ATEENA MVP 8.6.3.2

Nowości:
- overlay z odcinkami pomiarowymi na obrazach po analizie,
- komplet 4 zdjęć: FRONT, PROFIL LEWY, PROFIL PRAWY, TYŁ,
- rozszerzony standard landmarków o głowę i stopę (QA / auxiliary).

# ATEENA MVP 7.3

ATEENA MVP 7.3 rozwija wersję 7.2 o pięć najważniejszych warstw jakości produktu:

1. **Product Rescue / OCR fallback**
   - zrzut ekranu produktu,
   - zrzut ekranu tabeli rozmiarów,
   - OCR fallback dla produktu i tabeli,
   - automatyczne przejście na fallback kategorii, gdy OCR nie odczyta tabeli.

2. **QA / Admin panel**
   - liczniki użytkowników, analiz, produktów, rekomendacji, feedbacków,
   - rozkład quality gate,
   - najczęstsze komunikaty capture,
   - lista ostatnich sesji OCR,
   - lista zgłoszeń tłumaczeń.

3. **Benchmark**
   - syntetyczny benchmark heurystyczny na katalogu demo,
   - zapis historii benchmarków do bazy,
   - szybki odczyt zmian jakości logiki fitu.

4. **Split fit result**
   - osobny **fit techniczny**,
   - osobny **fit wizualny**,
   - czytelniejszy wynik końcowy dla użytkownika.

5. **Localization feedback loop**
   - formularz zgłaszania problemów z tłumaczeniem,
   - zapis locale, ekranu, tekstu i komentarza do bazy.

## Co pozostaje heurystyczne

- OCR fallback jest pomocnikiem MVP, nie pełnym parserem produkcyjnym.
- Benchmark jest syntetyczny i nie zastępuje walidacji na realnym secie testowym.
- Live capture overlay nadal nie jest pełnym custom mobile capture engine.

## Uruchomienie

```bash
pip install -r requirements.txt
streamlit run app.py
```

Baza SQLite dla tej wersji:

```text
data/ateena_mvp_v7_3.db
```


## Co nowego w v8.6.8

- zakładka **Capture Pro (beta)** z live helperem ustawienia FRONT / PROFIL,
- mocniejszy OCR fallback: screenshoty + ręcznie wklejony tekst OCR,
- benchmark na własnym secie CSV z metrykami MAE,
- dalsze spięcie QA i benchmarków z bazą.


## Nowości w v8.6.8
- mocniej dokręcony quality gate landmarków,
- rozdzielenie accept_ready / measurement_ready / posture_ready,
- lepsze komunikaty blockerów w Capture Pro,
- wyraźniejsza integracja sesji mobile z główną analizą.


## Nowości w v8.6.8
- pasma confidence per part (wysoki / średni / niski),
- wyraźniejsze zalecenia ręcznego potwierdzania słabych stref,
- bardziej konserwatywne traktowanie selfie i braku ujęcia TYŁ,
- lepsze rozdzielenie: pomiar vs screening postawy.


## Nowości w v8.6.8
- OCR fallback 2.0 z możliwością ręcznej korekty produktu i tabeli rozmiarów,
- Capture Pro 2.0: zapis sesji capture do QA/Admin,
- Calibration Loop 2.0 per part — zapis i podsumowanie kalibracji dla dodatkowych wymiarów,
- QA/Admin 2.0: ostatnie sesje Capture Pro oraz per-part calibration summary.


## Nowości w v8.6.8
- dołączony osobny pakiet `capture_mobile_pilot` do testów mobilnego capture przez HTTPS,
- nowa zakładka importu JSON z Mobile Pilot do głównej analizy,
- rozszerzony benchmark referencyjny z dodatkowymi partiami ciała,
- mocniejszy most między prototypem mobilnym a główną aplikacją.


## Nowości w v8.6.8
- Product Rescue 3.1: screenshot opinii / review OCR + ręczna korekta review text,
- visual search 3.1: ranking bierze pod uwagę wybraną kategorię produktu,
- import Mobile Pilot pokazuje skrócone podsumowanie jakości sesji,
- capture_mobile_pilot v2: auto-przejście FRONT → PROFIL → TYŁ i średni score sesji.


## Nowości w v8.6.8
- oczyszczanie zdjęć z tła przed analizą sylwetki,
- drugi pass landmarków na obrazie po cleanupie tła,
- score jakości odcięcia postaci od tła,
- podgląd klatek po oczyszczeniu tła,
- dodatkowy wpływ jakości tła na confidence i blockery capture.


## Nowości w v8.6.8
- Product Rescue 4.0: OCR quality score (produkt / tabela / review) oraz prostsza korekta tabeli przez edytor wierszy,
- Benchmark & Truth Set 3.0: rozszerzony szablon CSV, per-part summary, segment MAE, optional false accept / false reject,
- QA / Calibration 5.0: większa widoczność wyników benchmarku i kalibracji per part w panelu QA,
- Capture Mobile v3: utrzymany flow FRONT / PROFIL / TYŁ z eksportem JSON i lepszym mostem do głównej analizy.


## Hotfix 8.5 — OCR
- dodano brakującą zależność `pytesseract` do `requirements.txt`,
- OCR nie wywraca już aplikacji, jeśli `pytesseract` albo systemowy Tesseract nie są zainstalowane,
- na Windows aplikacja próbuje wykryć:
  - `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`

### Jeśli chcesz używać OCR na Windows
1. `pip install pytesseract`
2. zainstaluj systemowy Tesseract OCR
3. uruchom ponownie aplikację


## Nowości w v8.6.8
- w sekcji „Podgląd po oczyszczeniu tła” widać miejsca odczytu / estymacji pomiarów,
- linie pomiarowe są podpisane wartościami w cm,
- po analizie pokazywany jest zapisany zestaw wyników w cm.


## Nowości w v8.6.8
- integracja standardu landmarków ATEENA v1,
- osobne segmenty FRONT / PROFIL / TYŁ,
- lewa i prawa ręka / noga rozdzielone w raporcie segmentów,
- lepsze overlaye pomiarowe zgodne ze standardem anotacji.


## Nowości w v8.6.8
- nowa zakładka `Annotation Review`,
- import ręcznych adnotacji CSV,
- porównanie overlayu aplikacji z poprawną anotacją,
- scoring: visibility / occlusion / confidence accuracy,
- porównanie manualnych wartości cm z aktualnym body_result,
- zapis review adnotacji do bazy.


Nowości w 8.6.3:
- obsługa uploadu HEIC/HEIF/WEBP przez normalizację do JPEG,
- status każdego z 4 slotów zdjęć z podglądem,
- czytelniejszy komunikat o brakujących slotach,
- analiza poglądowa nawet gdy nie wszystkie ujęcia przechodzą ACCEPT,
- rekomendacja może zostać wygenerowana z ostrzeżeniem zamiast twardego stopa.


## Nowości w v8.6.8
- upload 4 zdjęć jest stabilizowany przez session_state,
- sloty FRONT/PROFIL LEWY/PROFIL PRAWY/TYŁ zachowują wczytane obrazy po rerunach,
- każdy slot pokazuje miniaturę, nazwę pliku i rozmiar,
- przycisk „Wyczyść wszystkie sloty zdjęć”,
- lepszy komunikat dla iPhone/Safari/Chrome mobilnego.


## Nowości w v8.6.8
- widoczny numer wersji i build ID w UI,
- pełniejszy panel statusu 4 slotów zdjęć: plik, format, rozmiar, wymiary, EXIF, payload,
- tryb diagnostyczny uploadu i analizy,
- eksport paczki debug sesji,
- analiza poglądowa mimo warningów quality gate,
- dokładniejszy komunikat o brakujących slotach.


## Nowości w v8.6.8
- wszystkie kluczowe pola startują puste,
- wymagane pola użytkownika i produktu mają placeholder `— wybierz —`,
- brak domyślnych danych wzrostu/wagi/wieku,
- ręczne wymiary nie są już prefillowane wartościami AI ani priorami,
- aplikacja wymusza świadome uzupełnienie wymaganych pól przed analizą i rekomendacją.


## Nowości w v8.6.8
- pola wymagane są oznaczone gwiazdką,
- dodano panel `Brakuje Ci jeszcze tych pól`,
- aplikacja pokazuje następne brakujące pole do uzupełnienia,
- zachowano puste startowe wartości dla wszystkich krytycznych pól.


## Nowości w v8.6.8
- runtime self-test dla vision pipeline,
- czerwony banner przy braku MediaPipe,
- rozróżnienie `photo-based estimate` vs `fallback prior only`,
- blokada generowania właściwej rekomendacji, gdy wynik pochodzi tylko z fallback prior,
- bardziej uczciwe komunikaty o źródle estymacji.
