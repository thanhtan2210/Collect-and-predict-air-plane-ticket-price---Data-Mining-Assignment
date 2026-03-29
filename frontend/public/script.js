document.getElementById('flightForm').addEventListener('submit', async function(e) {
    e.preventDefault(); // Ngăn load lại trang

    // Reset thông báo
    document.getElementById('result').innerText = '';
    document.getElementById('error').innerText = '';

    // Lấy dữ liệu
    const airline = document.getElementById('airline').value;
    const source = document.getElementById('source').value;
    const destination = document.getElementById('destination').value;
    const duration = document.getElementById('duration').value;
    const total_stops = document.getElementById('total_stops').value;

    // --- YÊU CẦU: VALIDATE DỮ LIỆU ---
    if (!airline || !source || !destination || !duration) {
        document.getElementById('error').innerText = '⚠️ Vui lòng điền đầy đủ thông tin chuyến bay!';
        return;
    }
    if (source === destination) {
        document.getElementById('error').innerText = '⚠️ Nơi đi và nơi đến không được trùng nhau!';
        return;
    }
    if (duration <= 0) {
        document.getElementById('error').innerText = '⚠️ Thời gian bay phải lớn hơn 0!';
        return;
    }

    // Chuẩn bị payload gửi xuống backend
    const payload = {
        airline: airline,
        source: source,
        destination: destination,
        duration: parseFloat(duration),
        total_stops: parseInt(total_stops)
    };

    // --- YÊU CẦU: GỬI DỮ LIỆU QUA API & XỬ LÝ LỖI ---
    try {
        const response = await fetch('http://127.0.0.1:8000/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok) {
            document.getElementById('result').innerText = `💰 Giá vé dự đoán: ${data.predicted_price} VNĐ`;
        } else {
            document.getElementById('error').innerText = `❌ Lỗi từ Server: ${data.detail}`;
        }
    } catch (error) {
        document.getElementById('error').innerText = '❌ Không thể kết nối tới Backend. Backend đã bật chưa?';
    }
});