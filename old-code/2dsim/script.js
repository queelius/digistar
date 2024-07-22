const canvas = document.getElementById('simulationCanvas');
const ctx = canvas.getContext('2d');

const scale = 2e9;  // Scale to fit the simulation into the canvas
const centerX = canvas.width / 2;
const centerY = canvas.height / 2;

const socket = new WebSocket('ws://localhost:8080');

socket.onopen = function() {
    console.log('WebSocket connection opened');
};

socket.onclose = function() {
    console.log('WebSocket connection closed');
};

socket.onerror = function(error) {
    console.log('WebSocket error: ' + error.message);
};

socket.onmessage = function(event) {
    const bodies = JSON.parse(event.data);
    render(bodies);
};

function render(bodies) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    bodies.forEach((body, index) => {
        const x = centerX + body.position.x / scale;
        const y = centerY - body.position.y / scale;  // Invert y-axis for correct orientation

        // Differentiate bodies by color
        ctx.beginPath();
        if (index === 0) {
            ctx.fillStyle = 'yellow';  // Sun
        } else if (index === 1) {
            ctx.fillStyle = 'blue';  // Earth
        } else if (index === 2) {
            ctx.fillStyle = 'gray';  // Moon
        }
        ctx.arc(x, y, 5, 0, 2 * Math.PI);  // Draw a small circle for each body
        ctx.fill();

        // Log positions for debugging
        if (index === 2) {  // Log moon's position
            console.log(`Moon position: x=${x}, y=${y}`);
        }
        if (x < 0 || x > canvas.width || y < 0 || y > canvas.height) {
            console.log(`Body ${index} is out of bounds: x=${x}, y=${y}`);
        }
    });
}
