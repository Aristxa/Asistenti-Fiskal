---
title: Asistenti Fiskal
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
license: mit
short_description: Pyetje-përgjigje me citime mbi legjislacionin tatimor
---

# Asistenti Fiskal

Bëj një pyetje shqip mbi detyrimet tatimore dhe merr një përgjigje ku **çdo pohim
është i lidhur me nenin përkatës** të legjislacionit — ose sistemi refuzon të përgjigjet.

Baza: **313 dokumente** zyrtare nga tri autoritete — Drejtoria e Përgjithshme e
Tatimeve, Këshilli Kombëtar i Kontabilitetit dhe Inspektorati Shtetëror i Punës —
të ndara në **20.615 copëza** sipas strukturës ligjore (neni / pika).

## Si funksionon

Kërkim hibrid mbi dy indekse: një semantik (dense) dhe një me fjalëkyçe (BM25),
të bashkuar me Reciprocal Rank Fusion. Mbi to, Claude përgjigjet nën një
**kontratë citimi**: çdo fjali faktike mbyllet me një referencë [S1], [S2] …,
ose sistemi thotë qartë se nuk e gjen përgjigjen.

### Shqipja, e trajtuar si duhet

Sistemi e kupton pyetjen edhe pa shkronjat `ë` dhe `ç` — sepse ashtu shkruan
shumica. Indeksi leksikor i njëson të dyja format, ndërsa pyetja rikthehet në
drejtshkrimin e saktë përpara kërkimit semantik. Gjithashtu njeh trajtat e
lakuara: *tatim, tatimi, tatimit, tatimin, tatime, tatimet, tatimeve* trajtohen
si e njëjta fjalë.

## Kufizimet

⚠️ **Ky nuk është këshillë tatimore.** Sistemi shpjegon çfarë thotë ligji dhe
tregon ku. Nuk llogarit detyrime konkrete dhe nuk zëvendëson kontabilistin.

⚠️ **Legjislacioni tatimor ndryshon shpesh.** Çdo përgjigje tregon vitin e aktit
që citon. Verifiko gjithmonë në burimin zyrtar përpara se të veprosh.

Pyetjet jashtë temës refuzohen — sistemi përgjigjet vetëm për legjislacionin
tatimor shqiptar.
