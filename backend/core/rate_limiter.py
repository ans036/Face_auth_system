"""
Rate Limiter Middleware for Face Auth System

Implements sliding window rate limiting to prevent:
- Brute force attacks on /identify/
- DoS attacks on the biometric endpoints
- Abuse of the authentication system

Features:
- IP-based rate limiting
- Configurable windows and limits
- In-memory storage (use Redis for production clusters)
- Exponential backoff for repeated violations
"""

import time
from collections import defaultdict
from typing import Dict, Tuple, Optional
from fastapi import Request, HTTPException
from dataclasses import dataclass, field
import threading


@dataclass
class RateLimitRecord:
    """Track request history for a single client."""
    requests: list = field(default_factory=list)  # Timestamps of requests
    blocked_until: float = 0  # Unix timestamp when block expires
    violation_count: int = 0  # Number of times limit exceeded


class RateLimiter:
    """
    Sliding window rate limiter with exponential backoff.
    
    Args:
        requests_per_minute: Maximum requests allowed per minute
        block_duration_seconds: Initial block duration after exceeding limit
        max_block_duration: Maximum block duration after repeated violations
        cleanup_interval: How often to clean up old records (seconds)
    """
    
    def __init__(
        self,
        requests_per_minute: int = 30,
        block_duration_seconds: int = 60,
        max_block_duration: int = 3600,  # 1 hour max
        cleanup_interval: int = 300  # 5 minutes
    ):
        self.requests_per_minute = requests_per_minute
        self.block_duration = block_duration_seconds
        self.max_block_duration = max_block_duration
        self.cleanup_interval = cleanup_interval
        
        self._records: Dict[str, RateLimitRecord] = defaultdict(RateLimitRecord)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        
        # Whitelist for trusted IPs (e.g., health checks, internal services)
        self.whitelist = {"127.0.0.1", "::1"}
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies."""
        # Check X-Forwarded-For header (if behind reverse proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # First IP in the chain is the original client
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_records(self, current_time: float):
        """Remove expired records to prevent memory growth."""
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        cutoff = current_time - 120  # Keep 2 minutes of history
        
        with self._lock:
            expired_keys = []
            for ip, record in self._records.items():
                # Remove old request timestamps
                record.requests = [t for t in record.requests if t > cutoff]
                
                # Mark for removal if no recent activity and not blocked
                if not record.requests and record.blocked_until < current_time:
                    expired_keys.append(ip)
            
            for key in expired_keys:
                del self._records[key]
            
            self._last_cleanup = current_time
    
    def _is_rate_limited(self, ip: str) -> Tuple[bool, Optional[int]]:
        """
        Check if IP is rate limited.
        
        Returns:
            (is_limited, retry_after_seconds)
        """
        current_time = time.time()
        self._cleanup_old_records(current_time)
        
        with self._lock:
            record = self._records[ip]
            
            # Check if currently blocked
            if record.blocked_until > current_time:
                retry_after = int(record.blocked_until - current_time)
                return True, retry_after
            
            # Clean old requests (outside 1-minute window)
            window_start = current_time - 60
            record.requests = [t for t in record.requests if t > window_start]
            
            # Check if over limit
            if len(record.requests) >= self.requests_per_minute:
                # Calculate block duration with exponential backoff
                backoff_multiplier = 2 ** record.violation_count
                block_seconds = min(
                    self.block_duration * backoff_multiplier,
                    self.max_block_duration
                )
                
                record.blocked_until = current_time + block_seconds
                record.violation_count += 1
                
                return True, block_seconds
            
            # Record this request
            record.requests.append(current_time)
            
            return False, None
    
    def add_to_whitelist(self, ip: str):
        """Add an IP to the whitelist."""
        self.whitelist.add(ip)
    
    def remove_from_whitelist(self, ip: str):
        """Remove an IP from the whitelist."""
        self.whitelist.discard(ip)
    
    async def check(self, request: Request) -> bool:
        """
        FastAPI dependency to check rate limit.
        
        Usage:
            @router.post("/")
            async def endpoint(request: Request, _: bool = Depends(rate_limiter.check)):
                ...
        """
        client_ip = self._get_client_ip(request)
        
        # Skip whitelist
        if client_ip in self.whitelist:
            return True
        
        is_limited, retry_after = self._is_rate_limited(client_ip)
        
        if is_limited:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        return True
    
    def get_status(self, ip: str) -> Dict:
        """Get rate limit status for an IP (for debugging/monitoring)."""
        current_time = time.time()
        
        with self._lock:
            record = self._records.get(ip)
            if not record:
                return {
                    "ip": ip,
                    "requests_in_window": 0,
                    "limit": self.requests_per_minute,
                    "blocked": False
                }
            
            window_start = current_time - 60
            recent_requests = [t for t in record.requests if t > window_start]
            
            return {
                "ip": ip,
                "requests_in_window": len(recent_requests),
                "limit": self.requests_per_minute,
                "blocked": record.blocked_until > current_time,
                "blocked_until": record.blocked_until if record.blocked_until > current_time else None,
                "violation_count": record.violation_count
            }


# Singleton instance for the identify endpoint
# Allow configuration via environment variables
import os

identify_rate_limiter = RateLimiter(
    requests_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", 120)),  # Boosted default to 120/min
    block_duration_seconds=int(os.getenv("RATE_LIMIT_BLOCK_SECONDS", 60)),
    max_block_duration=int(os.getenv("RATE_LIMIT_MAX_BLOCK_SECONDS", 3600))
)
