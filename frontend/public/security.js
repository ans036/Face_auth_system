const grid = document.getElementById('gridContainer');
const loadingText = document.getElementById('loadingText');
const apiBaseUrl = window.location.hostname === 'localhost' ? 'http://localhost:8001' : 'http://127.0.0.1:8001';

async function loadSnapshots() {
    try {
        const response = await fetch(`${apiBaseUrl}/security/snapshots/`);
        if (!response.ok) throw new Error("Failed to fetch records");

        const files = await response.json();
        loadingText.remove();

        if (files.length === 0) {
            grid.innerHTML = "<p>No unauthorized attempts recorded.</p>";
            return;
        }

        files.forEach(filename => {
            // Filename format: unknown_TIMESTAMP_score_SCORE.jpg
            const parts = filename.split('_');
            const timestamp = parseInt(parts[1]);
            const scoreRaw = parseInt(parts[3].split('.')[0]);

            // Format date nicely
            const date = new Date(timestamp * 1000);
            const dateStr = date.toLocaleString();

            createSnapshotCard(filename, dateStr, scoreRaw);
        });

    } catch (error) {
        console.error(error);
        loadingText.innerText = "Error loading security records. Backend offline?";
        loadingText.style.color = "var(--alert-color)";
    }
}

function createSnapshotCard(filename, dateStr, score) {
    const card = document.createElement('div');
    card.className = 'snapshot-card';

    const img = document.createElement('img');
    // Point to the backend API to serve the image
    img.src = `${apiBaseUrl}/security/snapshots/${filename}`;
    img.alt = "Unauthorized Subject";

    const info = document.createElement('div');
    info.className = 'snapshot-info';

    info.innerHTML = `
        <span class="timestamp">${dateStr}</span>
        <span class="score">Max Confidence: ${score}%</span>
    `;

    card.appendChild(img);
    card.appendChild(info);
    grid.appendChild(card);
}

// Load on startup
loadSnapshots();