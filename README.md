# LoRa na StableDiffusion


# Dodany Koncept

Projekt polega na nauczeniu modelu generowania zdjęć gumowej figurki ptaka inspirowanej grą angry birds. Wykorzystywany jest model Stable Diffusion w wersji 1.5.

# Dataset

Wykorzystany został samodzielnie przygotowany dataset, złożony z 20 zdjęć wykonanych telefonem komórkowym. 6 zdjęć przedstawia figurkę z różnych stron na białym tle, na pozostałych jest ona w kontekście różnych przedmiotów. Dataset nie zawiera zbliżeń na detale, obiektyw telefonu nie pozwalał na ich wykonanie ze względu na rozmiar figurki. Grafiki zostały poddane transformacjom resize oraz center crop, modyfikując ich wymiary do kwadratów o boku 512px.

| ![1](assets/image52.jpg)  |  ![2](assets/image4.jpg)  | ![3](assets/image27.jpg)  | ![4](assets/image47.jpg)  | ![5](assets/image25.jpg)  |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| ![6](assets/image23.jpg)  | ![7](assets/image44.jpg)  | ![8](assets/image17.jpg)  | ![9](assets/image33.jpg)  | ![10](assets/image19.jpg) |
| ![11](assets/image15.jpg) | ![12](assets/image55.jpg) | ![13](assets/image46.jpg) | ![14](assets/image14.jpg) | ![15](assets/image12.jpg) |
| ![11](assets/image5.jpg)  | ![12](assets/image42.jpg) | ![13](assets/image11.jpg) | ![14](assets/image39.jpg) | ![15](assets/image10.jpg) |
# LoRA

Proces fine-tuningu przeprowadzono metodą LoRA przy wykorzystaniu biblioteki peft. Trening trwał 1000 kroków z parametrami ustawionymi na rank=128 oraz alpha=128, przy lr=1e-4. Trenowane moduły: \["to\_q", "to\_k", "to\_v", "to\_out.0"\].  
W procesie wykorzystano prompty:

* INSTANCE\_PROMPT: „a photo of skp\_red\_bird figurine”  
* CLASS\_PROMPT: „a photo of small plastic figurine”

Celowo pominięta została informacja o podobieństwie figurki do postaci z gry komputerowej, zapobiegając mieszaniu informacji z wcześniej znanym konceptem.

Głównym problemem podczas treningu była pamięć laptopowej karty graficznej wynosząca 8GB. Program treningowy przed optymalizacją wykorzystywał ok. 11GB oraz jego wykonanie trwało ponad 25 minut w środowisku google colab. Dzięki przepuszczeniu zdjęć tylko raz, przed treningiem, przez VAE oraz text encoder a następnie usunięciu obiektów tych modeli z pamięci udało się zmniejszyć zapotrzebowanie na VRAM do ok 6GB. Dodatkowo karta RTX4070 laptop GPU oparta jest na nowszej technologii niż T4 dostępne za darmo na colabie. Po optymalizacji trening na 1000 epok trwa tylko 6 minut.

## Wygenerowane grafiki

Wszystkie grafiki generowane były przy 60 krokach interferencji, guidance\_scale=7.5 oraz negative\_prompt="deformed, blurred".

Prompt: "a photo of skp\_red\_bird figurine, underwater, at the bottom of the ocean, hyperrealistic"

<table>
  <tr>
    <td><img src="assets/image22.png" alt="1"></td>
    <td><img src="assets/image48.png" alt="2"></td>
    <td><img src="assets/image24.png" alt="3"></td>
    <td><img src="assets/image54.png" alt="4"></td>
    <td><img src="assets/image53.png" alt="5"></td>
  </tr>
</table>

Prompt:  "a photo of skp\_red\_bird figurine, from angry birds game,underwater, at the bottom of the ocean, hyperrealistic"

<table>
  <tr>
    <td><img src="assets/image3.png" alt="1"></td>
    <td><img src="assets/image50.png" alt="2"></td>
    <td><img src="assets/image28.png" alt="3"></td>
    <td><img src="assets/image35.png" alt="4"></td>
    <td><img src="assets/image38.png" alt="5"></td>
  </tr>
</table>

Prompt \= "a photo of bird figurine, from angry birds game,underwater, at the bottom of the ocean, hyperrealistic"

<table>
  <tr>
    <td><img src="assets/image31.png" alt="1"></td>
    <td><img src="assets/image7.png" alt="2"></td>
    <td><img src="assets/image49.png" alt="3"></td>
    <td><img src="assets/image20.png" alt="4"></td>
    <td><img src="assets/image32.png" alt="5"></td>
  </tr>
</table>

Warto zauważyć że w większości przypadków model generuje kilka figurek na jednym zdjęciu, pomimo braku takich instrukcji. W pierwszej próbie, bez informacji o grze, zdjęcia są najbardziej zdeformowane ale świetnie odwzorowywują fakturę gumy. W części wygenerowanych zdjęć usuwana jest podstawka, co ma sens w przypadku pływania pod wodą. W promptcie ze sformułowaniem “like in angry birds game” model często zapomina o umieszczeniu zabawki pod wodą, dalej świetnie oddając gumę. Przy prośbie o skp\_red\_bird z gry dalej generowany jest ptak bardzo podobny do rzeczywistego, z bardziej kreskówkowym tłem. W ostatnim, bez użycia tokenu dalej najczęściej występującą postacią jest czerwony ptak, najpopularniejszy w grze. Ma on bardzo dużo zmian kształtu lub koloru oraz zupełnie inne włosy.

## Fine Tuning LoRA bez warstwy "to\_out.0"

Powyższe zdjęcia zostały wygenerowane przy wykorzystaniu standardowego podejścia, trenując moduły atencji QKV oraz wyjściowego  "to\_out.0". Eksperymentalnie wykonano fine tuning bez modułu wyjściowego. Wszystkie pozostałe parametry treningu oraz generowania bez zmian.

Prompt:  "a photo of skp\_red\_bird figurine, like in angry birds game,underwater, at the bottom of the ocean, hyperrealistic"

<table>
  <tr>
    <td><img src="assets/image1.png" alt="1"></td>
    <td><img src="assets/image9.png" alt="2"></td>
    <td><img src="assets/image6.png" alt="3"></td>
    <td><img src="assets/image29.png" alt="4"></td>
    <td><img src="assets/image41.png" alt="5"></td>
  </tr>
</table> 

Pierwsza grafika wyszła prawie perfekcyjnie, w pozostałych widać znaczne zniekształcenia. Faktura dalej jest poprawnie odwzorowana ale model ma problemy z kształtem. 

## Generowanie nieznanego konceptu

Wewnątrz figurki znajduje się pendrive, wyjmowany wraz z podstawką. Nie została ta informacja podana w trakcie fine tuningu ani uwzględniona w datasecie. Chcemy sprawdzić wyobrażenia modelu. 

<table width="100%">
  <tr>
    <th width="20%">Zdjęcie rzeczywiste</th>
    <th colspan="4">Zdjęcia wygenerowane</th>
  </tr>
  <tr>
    <td width="20%"><img src="assets/image18.png" width="100%" alt="1"></td>
    <td width="20%"><img src="assets/image26.png" width="100%" alt="2"></td>
    <td width="20%"><img src="assets/image37.png" width="100%" alt="3"></td>
    <td width="20%"><img src="assets/image45.png" width="100%" alt="4"></td>
    <td width="20%"><img src="assets/image2.png" width="100%" alt="5"></td>
  </tr>
</table>

Wygenerowane grafiki są odległe od stanu rzeczywistego. Obudowa przypomina ptaki z innych testów, końcówka pendrive wygląda akceptowalnie, połączenie 2 obiektów jest nierealistyczne. 

# Textual Inversion

Trening przeprowadzono metodą Textual Inversion, zamrażając wagi UNet oraz VAE, optymalizując jedynie wektory embeddingów w przestrzeni Text Encodera. Proces trwał 2500 kroków przy z lr=5e-4. Token został zainicjowany słowem “red”, zgodnie z sugestią LLM po analizie zdjęcia z datasetu. Batch size wynosił 4\.

## Wygenerowane grafiki

Prompt \= "a photo of skp\_red\_bird like in angry birds game, underwater, at the bottom of the ocean, hyperrealistic", negative prompt="blur, low quality, distortion, ugly, bad anatomy"

<table>
  <tr>
    <td><img src="assets/image43.jpg" alt="1"></td>
    <td><img src="assets/image8.jpg" alt="2"></td>
    <td><img src="assets/image51.jpg" alt="3"></td>
    <td><img src="assets/image40.jpg" alt="4"></td>
    <td><img src="assets/image21.jpg" alt="5"></td>
  </tr>
</table>

Zdjęcia są widocznie bardziej zdeformowane niż w przypadku LoRA.