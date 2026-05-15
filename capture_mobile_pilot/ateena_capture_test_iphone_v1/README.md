# ATEENA Capture Test iPhone v1

To jest **osobny prototyp jakości wejścia** pod iPhone / Safari.
Nie liczy jeszcze finalnych obwodów i nie diagnozuje postawy.
Jego celem jest:

- uruchomić kamerę na telefonie,
- oceniać FRONT i PROFIL w czasie rzeczywistym,
- odrzucać złe ujęcia,
- zaakceptować tylko zdjęcia, które nadają się do dalszej analizy.

## Co sprawdza live

- głowa / twarz w kadrze,
- stopy w kadrze,
- pełna sylwetka,
- pozycja stojąca,
- front / profil / półprofil,
- czy ręce są lekko odsunięte,
- czy kamera nie jest przechylona,
- czy ubranie nie jest zbyt luźne.

## Statusy

- **ACCEPT** — można robić zdjęcie,
- **RETRY** — zdjęcie jest prawie poprawne, ale warto je poprawić,
- **REJECT** — zdjęcie nie nadaje się do analizy.

## Jak uruchomić

To jest statyczna aplikacja webowa. Najprościej wrzuć ją na:

- Vercel
- Netlify
- GitHub Pages + własny serwer HTTPS

albo uruchom lokalnie na komputerze i testuj na `localhost` przez tę samą sieć.

### Szybko lokalnie
Możesz użyć prostego serwera:
```bash
python -m http.server 8080
```

Potem wejdź na:
```text
http://localhost:8080
```

### Test na iPhonie
Kamera działa poprawnie tylko przez:
- **HTTPS**
- albo `localhost`

Na iPhonie otwórz stronę w **Safari** i daj dostęp do aparatu.

## Uwaga
Aby live pose działał, aplikacja używa:
- MediaPipe Tasks Vision przez CDN
- modelu Pose Landmarker przez internet

Czyli podczas testu telefon musi mieć dostęp do internetu.


## Nowości w 8.0
- dodatkowy krok TYŁ,
- prosty auto-capture po stabilnym ACCEPT,
- eksport JSON z FRONT / PROFIL / TYŁ.


## Pilot v2 w paczce 8.1
- automatyczne przechodzenie do kolejnego ujęcia po capture,
- podsumowanie jakości sesji,
- lepsze przygotowanie JSON do importu do głównej aplikacji.


## Dodatki v3
- podsumowanie quality sesji,
- auto-capture po stabilnym ACCEPT,
- eksport JSON przygotowany pod import do głównej aplikacji 8.3.
