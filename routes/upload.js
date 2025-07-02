// upload.js 
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

async function uploadAudio() {
  const form = new FormData();
  form.append("audio", fs.createReadStream("./sample.wav")); // Change to upload file

  try {
    const response = await axios.post("http://localhost:3000/upload", form, {
      headers: form.getHeaders(),
    });

    console.log("Server response:", response.data);
  } catch (err) {
    console.error("Upload failed:", err.message);
  }
}

uploadAudio();
