Projek ini memanfaatkan mediapipe landmarking untuk memproses gesture bahasa isyarat. Metode ini memiliki keterbatasan pada deteksi landmark oleh mediapipe itu sendiri, untuk mengatasi hal ini diperlukan standarisasi pada bagaimana pengambilan video seharusnya dilakukan untuk mencegah miss detection oleh mediapipe

# Contoh Miss Detection
![[Pasted image 20260325003311.png]]

Miss detection di atas terdapat pada face landmark pada gesture "Kamu" dengan latar belakang yang kompleks, hal ini mungkin disebabkan oleh muka yang tidak terbedakan secara jelas dari latar sehingga gagal dideteksi

catatan: kegagalan ini bisa jadi disebabkan oleh faktor lain yang belum diidentifikasi

# Latar Polos
![[Pasted image 20260325004002.png]]

latar belakang yang tidak kompleks bisa mencegah terjadinya miss detection
