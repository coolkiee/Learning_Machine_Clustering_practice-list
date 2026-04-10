import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dosyalar = ["s1.txt", "s2.txt", "s3.txt", "s4.txt"]
kume_sayisi = 15

print("--- Bölüm 1: s1, s2, s3, s4 Verilerinin K-Means ile Kümelenmesi ---")

for dosya in dosyalar:
    try:
        # Veriyi okuma
        df = pd.read_csv(dosya, sep=r'\s+', header=None)

        # --- VERİ TEMİZLİĞİ (DATA CLEANING) AŞAMASI ---
        # Sadece ilk 2 sütunu (X ve Y koordinatları) alıyoruz
        df = df.iloc[:, [0, 1]]

        # Sütunlardaki verileri zorla sayıya çeviriyoruz.
        # Eğer "it'son" gibi bir yazı varsa hata vermek yerine onu NaN (boş) yapar (errors='coerce')
        df[0] = pd.to_numeric(df[0], errors='coerce')
        df[1] = pd.to_numeric(df[1], errors='coerce')

        # İçinde harf olduğu için NaN (boş) olan o satırları veri setinden tamamen siliyoruz
        df = df.dropna()


        X = df[0].values
        y = df[1].values

        # K-Means modelini temizlenmiş verimizle eğitiyoruz
        kmeans = KMeans(n_clusters=kume_sayisi, init='k-means++', n_init=10, random_state=42)
        kume_etiketleri = kmeans.fit_predict(df)
        merkezler = kmeans.cluster_centers_

        # Grafik için figür oluşturma
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.canvas.manager.set_window_title(f'{dosya} Analizi')

        # Orijinal Veri Dağılımı
        ax1.scatter(X, y, s=10, color='gray')
        ax1.set_title(f"{dosya} - Orijinal Veri Dağılımı")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(alpha=0.3)

        # K-Means Sonucu ve Merkezler
        ax2.scatter(X, y, c=kume_etiketleri, cmap='tab20', s=10)
        ax2.scatter(merkezler[:, 0], merkezler[:, 1], c='red', marker='*', s=150, edgecolors='black', label='Centroids')
        ax2.set_title(f"{dosya} - K-Means Sonucu (K={kume_sayisi})")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

    except FileNotFoundError:
        print(f"HATA: {dosya} dosyası bulunamadı.")


# task 3: spiral.txt Analizi

print("\n--- Bölüm 2: spiral.txt Verisinin İncelenmesi ---")
dosya_spiral = "spiral.txt"

try:
    # Spiral verisini okuma
    df_spiral = pd.read_csv(dosya_spiral, sep=r'\s+', header=None)

    # Ne olur ne olmaz diye spiral dosyasına da veri temizliği uyguluyoruz
    df_spiral[0] = pd.to_numeric(df_spiral[0], errors='coerce')
    df_spiral[1] = pd.to_numeric(df_spiral[1], errors='coerce')
    df_spiral[2] = pd.to_numeric(df_spiral[2], errors='coerce')
    df_spiral = df_spiral.dropna()

    X_sp = df_spiral[0].values
    y_sp = df_spiral[1].values
    gercek_etiketler = df_spiral[2].values

    koordinatlar_spiral = df_spiral.iloc[:, [0, 1]]

    # K-Means ile 3 küme tahmini yapma
    kmeans_spiral = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    tahmin_edilen_etiketler = kmeans_spiral.fit_predict(koordinatlar_spiral)

    # Grafik için figür oluşturma
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title('Spiral Veri Seti Analizi')

    # Gerçek Kümeler
    ax1.scatter(X_sp, y_sp, c=gercek_etiketler, cmap='brg', s=15)
    ax1.set_title("Spiral Veri Seti - Gerçek Kümeler")
    ax1.grid(alpha=0.3)

    # K-Means Tahmini
    ax2.scatter(X_sp, y_sp, c=tahmin_edilen_etiketler, cmap='brg', s=15)
    ax2.set_title("Spiral Veri Seti - K-Means Tahmini")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

except FileNotFoundError:
    print(f"HATA: {dosya_spiral} dosyası bulunamadı.")

print("\nTüm hesaplamalar bitti. Bütün grafikler aynı anda açılıyor...")
plt.show()