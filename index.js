const express = require('express');
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');
const cors = require('cors');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' });

// Ensure upload and output directories exist
const uploadDir = path.join(__dirname, 'uploads');
const outputDir = path.join(__dirname, 'output');
const frontendDir = path.join(__dirname, 'frontend');

if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

// Middlewares
app.use(cors());
app.use(express.json());
app.use(express.static(frontendDir));  // Serve static files (index.html, CSS, JS)
app.use('/output', express.static(outputDir)); // Serve output files directly

// Root route
app.get('/', (req, res) => {
    res.sendFile(path.join(frontendDir, 'index.html'));
});

// Upload route
app.post('/upload', upload.single('audio'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    const inputPath = req.file.path;
    const baseFilename = path.parse(req.file.filename).name;
    const cleanedFilename = `${baseFilename}_cleaned.wav`;
    const transcriptFilename = `${baseFilename}_cleaned.txt`;

    const outputPath = path.join(outputDir, cleanedFilename);
    const scriptPath = path.join(__dirname, 'services', 'cleanAudio.py');

    console.log('Running script:', scriptPath);

    exec(`python3 "${scriptPath}" "${inputPath}" "${outputPath}"`, { cwd: __dirname }, (err, stdout, stderr) => {
        if (err) {
            console.error('❌ Script error:', stderr);
            return res.status(500).json({ error: 'Processing failed', details: stderr });
        }

        console.log('✅ Script output:\n', stdout);
        res.json({ 
            message: 'Audio cleaned and transcribed',
            downloadAudioUrl: `/output/${cleanedFilename}`,
            downloadTranscriptUrl: `/output/${transcriptFilename}`
        });
    });
});

// Start server
app.listen(3000, () => {
    console.log('🚀 Server running on http://localhost:3000');
});
