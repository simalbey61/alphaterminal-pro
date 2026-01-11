# 🚀 AlphaTerminal Pro - Ücretsiz Deployment Rehberi

## Gerekli Hesaplar (Hepsi Ücretsiz)

1. **GitHub** - github.com (kodları barındırmak için)
2. **Render** - render.com (backend için)
3. **Vercel** - vercel.com (frontend için)
4. **UptimeRobot** - uptimerobot.com (7/24 canlı tutmak için)

---

## 📦 ADIM 1: GitHub'a Yükle (5 dakika)

```bash
# 1. GitHub'da yeni repo oluştur: alphaterminal-pro

# 2. Terminalde:
cd alpha-terminal-pro
git init
git add .
git commit -m "Initial commit - AlphaTerminal Pro v1.0.0"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADIN/alphaterminal-pro.git
git push -u origin main
```

---

## 🔧 ADIM 2: Render'da Backend Deploy (10 dakika)

1. **render.com**'a git → GitHub ile giriş yap

2. **"New +"** → **"Web Service"** tıkla

3. GitHub reposunu bağla: `alphaterminal-pro`

4. Ayarları gir:
   ```
   Name:           alphaterminal-api
   Region:         Frankfurt (EU)
   Branch:         main
   Root Directory: backend
   Runtime:        Python 3
   Build Command:  pip install -r requirements.txt
   Start Command:  uvicorn app.main:app --host 0.0.0.0 --port $PORT
   Instance Type:  Free
   ```

5. **"Create Web Service"** tıkla

6. ⏳ 3-5 dakika bekle, deploy tamamlanacak

7. ✅ URL'ini not al: `https://alphaterminal-api.onrender.com`

### Test Et:
```
https://alphaterminal-api.onrender.com/api/v2/health
```
`{"status": "healthy"}` görmelisin.

---

## 🌐 ADIM 3: Vercel'de Frontend Deploy (10 dakika)

1. **vercel.com**'a git → GitHub ile giriş yap

2. **"Add New..."** → **"Project"** tıkla

3. GitHub reposunu seç: `alphaterminal-pro`

4. Ayarları gir:
   ```
   Framework Preset: Vite
   Root Directory:   frontend
   Build Command:    npm run build
   Output Directory: dist
   ```

5. **Environment Variables** ekle:
   ```
   VITE_API_URL = https://alphaterminal-api.onrender.com/api/v2
   ```

6. **"Deploy"** tıkla

7. ⏳ 2-3 dakika bekle

8. ✅ URL'ini not al: `https://alphaterminal-pro.vercel.app`

---

## ⏰ ADIM 4: UptimeRobot ile 7/24 Canlı Tut (5 dakika)

> ⚠️ Render free plan 15 dakika inaktivitede uyur. UptimeRobot bunu önler.

1. **uptimerobot.com**'a git → Ücretsiz kayıt ol

2. **"Add New Monitor"** tıkla

3. Ayarları gir:
   ```
   Monitor Type:       HTTP(s)
   Friendly Name:      AlphaTerminal API
   URL:                https://alphaterminal-api.onrender.com/api/v2/health
   Monitoring Interval: 5 minutes
   ```

4. **"Create Monitor"** tıkla

5. ✅ Artık backend hiç uyumayacak!

---

## 🎉 TAMAMLANDI!

### Senin URL'lerin:

| Servis | URL |
|--------|-----|
| **Frontend (Site)** | https://alphaterminal-pro.vercel.app |
| **Backend (API)** | https://alphaterminal-api.onrender.com |
| **API Docs** | https://alphaterminal-api.onrender.com/docs |
| **Health Check** | https://alphaterminal-api.onrender.com/api/v2/health |

---

## 🔄 Otomatik Güncellemeler

GitHub'a her push yaptığında:
- ✅ Render otomatik yeni backend deploy eder
- ✅ Vercel otomatik yeni frontend deploy eder

```bash
# Değişiklik yaptıktan sonra:
git add .
git commit -m "Yeni özellik eklendi"
git push origin main
# Otomatik deploy başlar!
```

---

## ❓ Sorun Giderme

### Backend açılmıyor?
1. Render Dashboard → Logs'a bak
2. Hata mesajını kontrol et
3. `requirements.txt` eksik paket olabilir

### Frontend API'ye bağlanamıyor?
1. Vercel → Settings → Environment Variables
2. `VITE_API_URL` doğru mu kontrol et
3. Redeploy yap

### Site yavaş açılıyor?
- İlk istek yavaş olabilir (cold start)
- UptimeRobot aktif mi kontrol et
- Sonraki istekler hızlı olacak

---

## 📱 Telegram Bildirimleri (Opsiyonel)

Render'da Environment Variables ekle:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 💡 İpuçları

1. **Custom Domain**: Vercel'de kendi domain'ini bağlayabilirsin
2. **SSL**: Otomatik ve ücretsiz (hem Render hem Vercel)
3. **Monitoring**: UptimeRobot'tan email/SMS alabilirsin
4. **Logs**: Her iki platformda da canlı log görüntüleme var

---

**Toplam Süre:** ~30 dakika
**Toplam Maliyet:** 0 TL 🎉
