"""
Performance Benchmarking Script for Face Auth System

This script measures and visualizes system performance metrics including:
- Face recognition latency
- Voice embedding latency
- Database query latency (SQLite vs PostgreSQL)
- End-to-end identification latency

Usage:
    python scripts/benchmark.py
    
Output:
    Generates publication-ready charts in scripts/output/
"""

import numpy as np
import time
import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# Ensure output directory exists
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_mock_embedding(dim: int = 512) -> np.ndarray:
    """Generate a random L2-normalized embedding."""
    emb = np.random.randn(dim).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)


def benchmark_face_embedding(n_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark face embedding generation.
    
    NOTE: This requires the InsightFace model to be loaded.
    Run inside Docker container or with models available.
    """
    try:
        from core.embedder import FaceEmbedder
        import cv2
        
        embedder = FaceEmbedder()
        
        # Create synthetic face image (112x112 RGB)
        test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            embedder.embed(test_image)
        
        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            embedder.embed(test_image)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
        }
    except ImportError as e:
        print(f"⚠️ Face embedding benchmark skipped: {e}")
        return {"error": str(e)}


def benchmark_voice_embedding(n_iterations: int = 50) -> Dict[str, float]:
    """
    Benchmark voice embedding generation.
    
    NOTE: Requires SpeechBrain model to be loaded.
    """
    try:
        from core.voice_embedder import get_voice_embedder
        import io
        import wave
        
        voice_embedder = get_voice_embedder()
        
        if not voice_embedder.is_available():
            return {"error": "Voice embedder not available"}
        
        # Create synthetic WAV file (1.5 seconds of random noise at 16kHz)
        sample_rate = 16000
        duration = 1.5
        audio_data = (np.random.randn(int(sample_rate * duration)) * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        audio_bytes = buffer.getvalue()
        
        # Warmup
        for _ in range(3):
            voice_embedder.embed_from_bytes(audio_bytes, format="wav")
        
        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            voice_embedder.embed_from_bytes(audio_bytes, format="wav")
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
        }
    except ImportError as e:
        print(f"⚠️ Voice embedding benchmark skipped: {e}")
        return {"error": str(e)}


def benchmark_database_query(gallery_sizes: List[int] = [10, 100, 500, 1000]) -> Dict[str, List[float]]:
    """
    Benchmark database similarity search at various gallery sizes.
    
    Compares in-memory NumPy search vs pgvector (if available).
    """
    results = {
        "gallery_sizes": gallery_sizes,
        "numpy_ms": [],
        "pgvector_ms": [],
    }
    
    for size in gallery_sizes:
        print(f"  Testing gallery size: {size}")
        
        # Generate mock gallery
        gallery_matrix = np.vstack([generate_mock_embedding(512) for _ in range(size)])
        probe = generate_mock_embedding(512)
        
        # Benchmark NumPy vectorized search
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            scores = gallery_matrix @ probe  # Vectorized dot product
            top_idx = np.argmax(scores)
            latencies.append((time.perf_counter() - start) * 1000)
        
        results["numpy_ms"].append(np.mean(latencies))
        
        # Benchmark pgvector (if available)
        try:
            from db.models import USE_PGVECTOR
            from db.crud import search_similar_faces
            
            if USE_PGVECTOR:
                latencies = []
                for _ in range(20):  # Fewer iterations for DB
                    start = time.perf_counter()
                    search_similar_faces(probe, limit=10)
                    latencies.append((time.perf_counter() - start) * 1000)
                results["pgvector_ms"].append(np.mean(latencies))
            else:
                results["pgvector_ms"].append(None)
        except Exception as e:
            print(f"    pgvector not available: {e}")
            results["pgvector_ms"].append(None)
    
    return results


def plot_database_comparison(results: Dict, output_path: Path):
    """Generate bar chart comparing SQLite vs PostgreSQL performance."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results["gallery_sizes"]))
        width = 0.35
        
        # NumPy bars
        ax.bar(x - width/2, results["numpy_ms"], width, label='NumPy (In-Memory)', color='#3498db')
        
        # pgvector bars (if available)
        pgvector_vals = [v if v is not None else 0 for v in results["pgvector_ms"]]
        if any(pgvector_vals):
            ax.bar(x + width/2, pgvector_vals, width, label='pgvector (PostgreSQL)', color='#2ecc71')
        
        ax.set_xlabel('Gallery Size (Number of Embeddings)', fontsize=12)
        ax.set_ylabel('Query Latency (ms)', fontsize=12)
        ax.set_title('Face Recognition Search Performance: SQLite vs PostgreSQL', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(results["gallery_sizes"])
        ax.legend()
        
        # Add value labels
        for i, v in enumerate(results["numpy_ms"]):
            ax.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_path}")
        
    except ImportError:
        print("⚠️ matplotlib/seaborn not available, skipping plot generation")


def plot_latency_distribution(latencies: List[float], title: str, output_path: Path):
    """Generate latency distribution histogram."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.histplot(latencies, bins=30, kde=True, ax=ax, color='#3498db')
        
        # Add percentile lines
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        ax.axvline(p50, color='green', linestyle='--', label=f'P50: {p50:.1f}ms')
        ax.axvline(p95, color='orange', linestyle='--', label=f'P95: {p95:.1f}ms')
        ax.axvline(p99, color='red', linestyle='--', label=f'P99: {p99:.1f}ms')
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_path}")
        
    except ImportError:
        print("⚠️ matplotlib/seaborn not available, skipping plot generation")


def plot_scalability_curve(results: Dict, output_path: Path):
    """Generate line chart showing query latency vs gallery size for scalability analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sizes = results["gallery_sizes"]
        
        # Plot NumPy line
        ax.plot(sizes, results["numpy_ms"], 'o-', color='#3498db', 
                linewidth=2, markersize=8, label='NumPy (In-Memory)')
        
        # Plot pgvector line (if available)
        pgvector_vals = [v for v in results["pgvector_ms"] if v is not None]
        if pgvector_vals:
            pgvector_sizes = [s for s, v in zip(sizes, results["pgvector_ms"]) if v is not None]
            ax.plot(pgvector_sizes, pgvector_vals, 's-', color='#2ecc71', 
                    linewidth=2, markersize=8, label='pgvector (HNSW)')
        
        # Add O(n) reference line
        scale = results["numpy_ms"][0] / sizes[0]
        linear_ref = [s * scale for s in sizes]
        ax.plot(sizes, linear_ref, '--', color='gray', alpha=0.5, label='O(n) reference')
        
        ax.set_xlabel('Gallery Size (Number of Embeddings)', fontsize=12)
        ax.set_ylabel('Query Latency (ms)', fontsize=12)
        ax.set_title('Scalability Analysis: Query Latency vs Gallery Size', fontsize=14)
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add grid
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_path}")
        
    except ImportError:
        print("⚠️ matplotlib/seaborn not available, skipping plot generation")


def generate_summary_table(face_results: Dict, voice_results: Dict, db_results: Dict) -> str:
    """Generate Markdown summary table."""
    table = """
# Performance Benchmark Results

## Embedding Generation Latency

| Metric | Face (512-dim) | Voice (192-dim) |
|--------|----------------|-----------------|
"""
    
    if "error" not in face_results:
        table += f"| Mean | {face_results['mean_ms']:.2f} ms | "
    else:
        table += f"| Mean | N/A ({face_results['error']}) | "
    
    if "error" not in voice_results:
        table += f"{voice_results['mean_ms']:.2f} ms |\n"
        table += f"| P50 | {face_results.get('p50_ms', 'N/A'):.2f} ms | {voice_results['p50_ms']:.2f} ms |\n"
        table += f"| P95 | {face_results.get('p95_ms', 'N/A'):.2f} ms | {voice_results['p95_ms']:.2f} ms |\n"
        table += f"| P99 | {face_results.get('p99_ms', 'N/A'):.2f} ms | {voice_results['p99_ms']:.2f} ms |\n"
    else:
        table += f"N/A ({voice_results['error']}) |\n"
    
    table += """
## Database Query Latency

| Gallery Size | NumPy (ms) | pgvector (ms) | Speedup |
|--------------|------------|---------------|---------|
"""
    
    for i, size in enumerate(db_results["gallery_sizes"]):
        numpy_ms = db_results["numpy_ms"][i]
        pgv_ms = db_results["pgvector_ms"][i]
        if pgv_ms:
            speedup = f"{numpy_ms / pgv_ms:.1f}x"
            table += f"| {size} | {numpy_ms:.2f} | {pgv_ms:.2f} | {speedup} |\n"
        else:
            table += f"| {size} | {numpy_ms:.2f} | N/A | - |\n"
    
    return table


def main():
    print("=" * 60)
    print("Face Auth System - Performance Benchmark")
    print("=" * 60)
    
    # 1. Face embedding benchmark
    print("\n📸 Benchmarking face embedding generation...")
    face_results = benchmark_face_embedding(n_iterations=100)
    if "error" not in face_results:
        print(f"   Mean latency: {face_results['mean_ms']:.2f} ms (P99: {face_results['p99_ms']:.2f} ms)")
    
    # 2. Voice embedding benchmark
    print("\n🎤 Benchmarking voice embedding generation...")
    voice_results = benchmark_voice_embedding(n_iterations=50)
    if "error" not in voice_results:
        print(f"   Mean latency: {voice_results['mean_ms']:.2f} ms (P99: {voice_results['p99_ms']:.2f} ms)")
    
    # 3. Database query benchmark
    print("\n🔍 Benchmarking database similarity search...")
    db_results = benchmark_database_query(gallery_sizes=[10, 50, 100, 500, 1000])
    
    # 4. Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_database_comparison(db_results, OUTPUT_DIR / "db_comparison.png")
    plot_scalability_curve(db_results, OUTPUT_DIR / "scalability_curve.png")
    
    # 5. Generate summary report
    print("\n📝 Generating summary report...")
    summary = generate_summary_table(face_results, voice_results, db_results)
    summary_path = OUTPUT_DIR / "benchmark_report.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"   Saved: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ Benchmark complete! Results in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
