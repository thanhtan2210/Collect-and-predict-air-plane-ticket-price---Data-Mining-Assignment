const express = require('express');
const path = require('path');

const app = express();
const PORT = 3000;

// Phục vụ các file tĩnh (HTML, CSS, JS) trong thư mục 'public'
app.use(express.static(path.join(__dirname, 'public')));

// Trả về file index.html khi người dùng vào trang chủ
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Frontend server đang chạy tại: http://localhost:${PORT}`);
});