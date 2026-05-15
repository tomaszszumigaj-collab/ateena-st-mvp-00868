ATEENA MVP 8.6.7.2

Nowości:
- overlay z odcinkami pomiarowymi na obrazach po analizie,
- komplet 4 zdjęć: FRONT, PROFIL LEWY, PROFIL PRAWY, TYŁ,
- rozszerzony standard landmarków o głowę i stopę (QA / auxiliary).

ATEENA MVP 8.0

Co nowego:
- Capture Pro (beta): live helper ustawienia FRONT / PROFIL z kamerą i podstawowym quality gate.
- OCR fallback rozszerzony o screenshot produktu, screenshot tabeli oraz ręcznie wklejony tekst OCR.
- Benchmark na własnym secie CSV (MAE dla biustu / talii / bioder) + szablon CSV.
- Wersja bazy danych: data/ateena_mvp_v7_4.db

Uruchomienie:
1. Wejdź do folderu ATEENA_MVP_7_4/ATEENA_MVP_7_4
2. pip install -r requirements.txt
3. streamlit run app.py

Uwaga:
- Capture Pro jest helperem beta i nie zapisuje jeszcze zdjęć do głównego pipeline analizy.
- Pack lokalizacyjny został dołączony bez zmian z wcześniejszej wersji.


Nowość: Capture Pro zapisuje sesję do st.session_state i może zostać użyty bezpośrednio w głównej analizie.

Nowości w 8.0:
- ostrzejszy quality gate landmarków i accept/reject logic
- measurement_ready vs posture_ready dla sesji Capture Pro
- więcej komunikatów blokujących przy słabych zdjęciach
- poprawiona integracja Capture Pro z główną analizą


Nowości w 8.0:
- confidence bands dla każdej partii ciała
- capture action plan z blokadami jakości
- ostrzejsze traktowanie selfie i braku TYŁU
- bardziej konserwatywne weak points


Nowości w 8.0:
- OCR fallback 2.0 z ręczną korektą wyniku OCR
- Capture Pro 2.0 zapisuje sesje do QA/Admin
- Calibration Loop 2.0 per part dla dodatkowych wymiarów
- QA/Admin 2.0 pokazuje recent capture sessions i part calibration summary


Nowości w 8.0:
- folder capture_mobile_pilot w paczce
- import JSON z mobile capture do głównej analizy
- rozszerzony benchmark referencyjny CSV
- przygotowanie pod prawdziwy mobile capture flow


Nowości w 8.6.7:
- Product Rescue 3.1 z review OCR
- visual search 3.1 z category-aware ranking
- import Mobile Pilot z podsumowaniem jakości sesji
- capture_mobile_pilot v2 z auto-przejściem FRONT/PROFIL/TYŁ


Nowości w 8.6.7:
- background removal / cleanup tła przed analizą
- podgląd klatek po oczyszczeniu tła
- cleanup score wpływa na confidence i quality gate


Nowości w 8.6.7:
- Product Rescue 4.0 z OCR quality score i edytorem tabeli
- Benchmark & Truth Set 3.0 z segment MAE i false accept/reject
- QA / Calibration 5.0 z lepszym widokiem per-part
- Mobile capture v3 jako most do głównej analizy


Hotfix 8.6.7:
- dodano pytesseract do requirements
- OCR stał się opcjonalny i nie blokuje startu aplikacji


Nowości w 8.6.7:
- wizualizacja miejsc odczytu pomiarów na oczyszczonych obrazach,
- podpisanie linii pomiarowych wartościami w cm,
- lepszy podgląd dla zdjęć dodawanych z dysku.


Nowości w 8.6.7:
- integracja standardu landmarków ATEENA v1,
- raport segmentów landmarków,
- lepsze overlaye pomiarowe zgodne ze standardem FRONT/PROFIL/TYŁ.


Nowości w 8.6.7:
- Annotation Review
- import CSV adnotacji
- scoring overlay vs anotacja
- zapis review do bazy


Nowości w 8.6.7:
- obsługa uploadu HEIC/HEIF/WEBP przez normalizację do JPEG,
- status każdego z 4 slotów zdjęć z podglądem,
- czytelniejszy komunikat o brakujących slotach,
- analiza poglądowa nawet gdy nie wszystkie ujęcia przechodzą ACCEPT,
- rekomendacja może zostać wygenerowana z ostrzeżeniem zamiast twardego stopa.


Nowości w 8.6.7:
- stabilny upload 4 zdjęć na mobile i desktop
- session_state dla slotów uploadu
- debug nazw plików i rozmiarów


Nowości w 8.6.7:
- build/version visible
- diagnostics mode
- exact missing slots
- debug bundle export
- preview analysis despite warnings


Nowości w 8.6.7:
- brak prefilli w polach krytycznych
- required placeholders dla user/product flow
- ręczne wymiary startują puste
- walidacja przed analizą i rekomendacją


Nowości w 8.6.7:
- oznaczenie pól obowiązkowych gwiazdką
- panel brakujących pól
- wskazanie następnego brakującego pola


Nowości w 8.6.7:
- runtime self-test
- source label photo-based vs fallback prior only
- blokada mylącego fallbacku przy rekomendacji
