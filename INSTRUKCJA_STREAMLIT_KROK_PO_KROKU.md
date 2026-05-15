# ATEENA MVP 8.6.8 — instrukcja krok po kroku: GitHub + Streamlit Community Cloud

## 0. Co jest w tej paczce
Ta paczka jest przygotowana do wdrożenia na Streamlit Community Cloud.
Najważniejsze pliki w katalogu głównym:
- `app.py`
- `requirements.txt`
- `packages.txt`
- foldery `assets`, `data`, `pack`, `capture_mobile_pilot`

## 1. Rozpakuj ZIP
Rozpakuj ten ZIP na komputerze.
Nie wrzucaj ZIP-a jako ZIP do GitHub — wrzucasz **rozpakowane pliki**.

## 2. Utwórz nowe repo na GitHub
1. Zaloguj się do GitHub.
2. Kliknij `+` w prawym górnym rogu.
3. Kliknij `New repository`.
4. Ustaw:
   - Repository name: `ateena-mvp-868`
   - Visibility: `Public`
   - NIE dodawaj README
   - NIE dodawaj .gitignore
   - NIE dodawaj license
5. Kliknij `Create repository`.

## 3. Wgraj pliki do GitHub
1. W nowym repo kliknij `Add file` → `Upload files`.
2. Przeciągnij do przeglądarki **zawartość rozpakowanego folderu**, tak aby `app.py` był w root repo.
3. Na dole wpisz commit message, np. `Initial upload of ATEENA MVP 8.6.8`.
4. Kliknij `Commit changes`.

## 4. Utwórz aplikację w Streamlit Community Cloud
1. Wejdź na `share.streamlit.io`.
2. Zaloguj się i połącz GitHub.
3. Kliknij `Create app`.
4. Wybierz:
   - Repository: Twoje nowe repo
   - Branch: `main`
   - Main file path: `app.py`

## 5. Advanced settings — najważniejsze
Kliknij `Advanced settings` i ustaw:
- Python version: **3.12**

To jest kluczowe dla MediaPipe.

## 6. Deploy
Kliknij `Deploy` i poczekaj na build.

## 7. Po uruchomieniu sprawdź 3 rzeczy
1. Czy na górze aplikacji widać wersję `8.6.8`.
2. Czy runtime self-test pokazuje `MediaPipe: OK`.
3. Czy aplikacja ładuje się bez błędu `Error installing requirements`.

## 8. Jeśli build się wywali
1. Otwórz aplikację.
2. Kliknij `Manage app` w prawym dolnym rogu.
3. Otwórz `Cloud logs`.
4. Skopiuj pierwsze 30–50 linii błędu.

## 9. Jeśli wcześniej wdrożyłeś starą wersję
Najbezpieczniej trzymaj **nowe repo i nową appkę** dla 8.6.8.
Dopiero gdy wszystko działa, możesz usunąć starą appkę.

## 10. Co testować po wdrożeniu
- upload 4 zdjęć z iPhone'a
- runtime self-test
- status slotów zdjęć
- analyse body
- source label: `photo-based estimate` vs `fallback prior only`
