import streamlit as st
import datetime
import pandas as pd
import plotly.express as px

# Custom Header
st.markdown(
    """
    <style>
    .header {
        background-color: #D6D6F5;
        padding: 10px;
        text-align: center;
        border-radius: 7px;
    }
    .header img {
        width: 60px;
    }
    </style>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/id/2/2d/Undip.png" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit Title
st.markdown(
    """
    <h1 style="text-align: center;">UV Index</h1>
    """,
    unsafe_allow_html=True,
)

# Simulasi Data Dummy
latest_time = datetime.datetime.now()
uv_index = 5  # Nilai UV Index tetap (bisa diubah sesuai keinginan)

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=uv_index,
    gauge={
        'axis': {'range': [0, 11]},
        'bar': {'color': "#3098ff"},
        'steps': [
            {'range': [0, 3], 'color': "#00ff00"},
            {'range': [3, 6], 'color': "#ffff00"},
            {'range': [6, 8], 'color': "#ff6600"},
            {'range': [8, 10], 'color': "#ff0000"},
            {'range': [10,11], 'color': "#9900cc"},
        ]
    }
))

fig.update_layout(
    margin=dict(t=30, b=30, l=30, r=30),
)

st.plotly_chart(fig, use_container_width=True)

# Menambahkan widget himbauan
st.markdown(
    f"""
    <div style="text-align: center;">
        <span style="display: inline-block; padding: 5px 15px; border-radius: 5px;
                    background-color: {'#d4edda' if uv_index <= 2 else '#fcfac0' if uv_index <= 5 else '#ffc78f' if uv_index <= 7 else '#ff8a8a' if uv_index <= 10 else '#e7cafc'};">
            {"<p style='color: #00ff00;'><strong>✅ Tingkat aman:</strong> Gunakan pelembab tabir surya SPF 30+ dan kacamata hitam.</p>" if uv_index <= 2 else
             "<p style='color: #ffcc00;'><strong>⚠️ Tingkat bahaya sedang:</strong> Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam, kenakan pakaian pelindung matahari.</p>" if uv_index <= 5 else
             "<p style='color: #ff6600;'><strong>⚠️ Tingkat bahaya tinggi:</strong> Kurangi paparan matahari antara pukul 10 pagi hingga pukul 4 sore.</p>" if uv_index <= 7 else
             "<p style='color: #ff0000;'><strong>⚠️ Tingkat bahaya sangat tinggi:</strong> Tetap di tempat teduh dan oleskan sunscreen setiap 2 jam.</p>" if uv_index <= 10 else
             "<p style='color: #9900cc;'><strong>❗ Tingkat bahaya ekstrem:</strong> Diperlukan semua tindakan pencegahan karena kulit dan mata dapat rusak dalam hitungan menit.</p>"}
       </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# Simulasi Data Prediksi Dummy
future_df = pd.DataFrame({
    "Time": [latest_time + pd.Timedelta(minutes=30 * i) for i in range(5)],
    "Predicted Index": [4, 6, 8, 9, 11]  # Data dummy
})

# Menampilkan UV Index saat ini
st.metric(label="Current UV Index", value=uv_index)

# Warna indikator UV berdasarkan level
def get_uv_color(index):
    if index <= 2:
        return "green"
    elif index <= 5:
        return "yellow"
    elif index <= 7:
        return "orange"
    elif index <= 10:
        return "red"
    else:
        return "purple"

# Kotak indikator UV
st.markdown(
    f"""
    <div style="padding: 20px; text-align: center; background-color: {get_uv_color(uv_index)}; color: white; font-size: 24px; border-radius: 10px;">
        UV Index: {uv_index}
    </div>
    """,
    unsafe_allow_html=True
)

# Grafik Prediksi UV Index
fig = px.line(future_df, x="Time", y="Predicted Index", markers=True, title="Predicted UV Index")
st.plotly_chart(fig)

# Catatan Keselamatan Berdasarkan UV Index
if uv_index <= 2:
    st.success("UV rendah. Aman untuk aktivitas luar ruangan.")
elif uv_index <= 5:
    st.warning("UV sedang. Gunakan topi dan kacamata hitam.")
elif uv_index <= 7:
    st.warning("UV tinggi. Pakai sunscreen dan cari tempat teduh.")
elif uv_index <= 10:
    st.error("UV sangat tinggi. Hindari terlalu lama di bawah matahari!")
else:
    st.error("UV ekstrem! Batasi waktu di luar ruangan sebisa mungkin.")

# Jalankan dengan: streamlit run streamlit_app.py

# Menambahkan tabel saran proteksi
st.markdown(
    """
    <h1 style="text-align: center;margin-top: 40px; margin-bottom: 10px;">Tabel Saran Proteksi</h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <table style="width:100%; border-collapse: collapse; text-align: center;">
        <tr>
            <th style="border: 1px solid black; padding: 8px;">Kategori</th>
            <th style="border: 1px solid black; padding: 8px;">Himbauan</th>
        </tr>
        <tr style="background-color: #00ff00;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">0-2 (Low)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya rendah bagi orang banyak.</li>
                    <li>Kenakan kacamata hitam pada hari yang cerah.</li>
                    <li>Gunakan cairan pelembab tabir surya SPF 30+ bagi kulit sensitif.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
        <tr style="background-color: #ffff00;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">3-5 (Moderate)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya sedang bagi orang yang terpapar matahari tanpa pelindung.</li>
                    <li>Tetap di tempat teduh pada saat matahari terik siang hari.</li>
                    <li>Kenakan pakaian pelindung matahari, topi lebar, dan kacamata hitam yang menghalangi sinar UV, pada saat berada di luar ruangan.</li>
                    <li>Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam bahkan pada hari berawan, setelah berenang atau berkeringat.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
        <tr style="background-color: #ff6600;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">6-7 (High)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya tinggi bagi orang yang terpapar matahari tanpa pelindung, diperlukan pelindung untuk menghindari kerusakan mata dan kulit.</li>
                    <li>Kurangi waktu di bawah paparan matahari antara pukul 10 pagi hingga pukul 4 sore.</li>
                    <li>Kenakan pakaian pelindung matahari, topi lebar, dan kacamata hitam yang menghalangi sinar UV, pada saat berada di luar ruangan.</li>
                    <li>Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam bahkan pada hari berawan, setelah berenang atau berkeringat.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
        <tr style="background-color: #ff0000;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">8-10 (Very High)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya tinggi bagi orang yang terpapar matahari tanpa pelindung, diperlukan pelindung untuk menghindari kerusakan mata dan kulit.</li>
                    <li>Minimal waktu di bawah paparan matahari antara pukul 10 pagi hingga pukul 4 sore.</li>
                    <li>Tetap di tempat teduh pada saat matahari terik siang hari.</li>
                    <li>Kenakan pakaian pelindung matahari, topi lebar, dan kacamata hitam yang menghalangi sinar UV, pada saat berada di luar ruangan.</li>
                    <li>Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam bahkan pada hari berawan, setelah berenang atau berkeringat.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
        <tr style="background-color: #9900cc;">
            <td style="border: 1px solid black; padding: 8px; text-align: left;">11+ (Extreme)</td>
            <td style="border: 1px solid black; padding: 8px; text-align: left;">
                <ul>
                    <li>Tingkat bahaya ekstrem, diperlukan semua tindakan pencegahan karena kulit dan mata dapat rusak dalam hitungan menit.</li>
                    <li>Hindari paparan matahari langsung dan pastikan perlindungan maksimal.</li>
                    <li>Tetap di tempat teduh pada saat matahari terik siang hari.</li>
                    <li>Kenakan pakaian pelindung matahari, topi lebar, dan kacamata hitam yang menghalangi sinar UV, pada saat berada di luar ruangan.</li>
                    <li>Oleskan cairan pelembab tabir surya SPF 30+ setiap 2 jam bahkan pada hari berawan, setelah berenang atau berkeringat.</li>
                    <li>Permukaan yang cerah, seperti pasir, air, dan salju, akan meningkatkan paparan UV.</li>
                </ul>
            </td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True,
)

# Custom Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 70px;
        font-size: 12px;
        text-align: left;
        margin: 0;
        padding: 5px 10px;
    }
    </style>
    <div class="footer">
        <p>Universitas Diponegoro<br>Fakultas Sains dan Matematika<br>Departemen Fisika</p>
        <p>Nastangini<br>20440102130112</p>
    </div>
    """,
    unsafe_allow_html=True
)
