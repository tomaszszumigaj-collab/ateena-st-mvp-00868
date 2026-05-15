ATEENA MVP 8.6.2.2

Nowości:
- overlay z odcinkami pomiarowymi na obrazach po analizie,
- komplet 4 zdjęć: FRONT, PROFIL LEWY, PROFIL PRAWY, TYŁ,
- rozszerzony standard landmarków o głowę i stopę (QA / auxiliary).

Uruchamiaj z tego folderu:

1. python -m venv .venv
2. aktywuj środowisko
3. pip install -r requirements.txt
4. streamlit run app.py

Jeśli chcesz używać OCR:
- pip install pytesseract
- zainstaluj systemowy Tesseract OCR

Aplikacja uruchomi się także bez OCR, ale screenshoty tabel i opinii nie będą wtedy działały.

Wersja 8.5: przy zdjęciach dodawanych z dysku zobaczysz na obrazie po oczyszczeniu tła, gdzie zostały odczytane / oszacowane pomiary i jakie wartości w cm zapisano.

Wersja 8.5: aplikacja integruje standard landmarków ATEENA v1 i pokazuje raport segmentów FRONT/PROFIL/TYŁ oraz lepsze overlaye zgodne ze standardem.


Nowości w 8.6.2:
- obsługa uploadu HEIC/HEIF/WEBP przez normalizację do JPEG,
- status każdego z 4 slotów zdjęć z podglądem,
- czytelniejszy komunikat o brakujących slotach,
- analiza poglądowa nawet gdy nie wszystkie ujęcia przechodzą ACCEPT,
- rekomendacja może zostać wygenerowana z ostrzeżeniem zamiast twardego stopa.

Wersja 8.6.4 skupia się na stabilizacji uploadu 4 zdjęć i diagnostyce jakości.
