const form = document.getElementById('uploadForm');
const responseDiv = document.getElementById('response');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    responseDiv.textContent = 'Uploading...';

    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) {
        responseDiv.textContent = 'Please select a file first.';
        return;
    }

    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);

    try {
        const res = await fetch('http://localhost:3000/upload', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            throw new Error(`Server error: ${res.statusText}`);
        }

        const data = await res.json();
        responseDiv.innerHTML = `
  ✅ File cleaned successfully!<br>
  <a href="${data.downloadUrl}" download>⬇️ Download Cleaned File</a>
`;
    } catch (err) {
        responseDiv.textContent = `Error: ${err.message}`;
    }
});
