# Demo Inferență NUC-Net & Alpine

Acest director conține notebook-uri interactive Jupyter care demonstrează inferența pentru segmentarea panoptică LiDAR folosind modelul distilat NUC-Net și head-ul de clusterizare Alpine.

Există două versiuni ale notebook-ului, în funcție de mediul în care se dorește rularea codului.

---

## 1. Demo Inferență Locală (`local_inference_demo.ipynb`)

Acest notebook este configurat pentru a rula local, folosind fișierele stocate în acest repositoriu.

### Instrucțiuni de Configurare

1. **Configurarea Mediului**:
   Notebook-ul necesită biblioteci specifice precum `spconv-cu11x` și `torch-scatter`. Mediul necesar se poate configura ușor folosind fișierul de mediu Conda localizat în rădăcina proiectului:
   ```bash
   cd ..
   conda env create -f nucnet_environment.yml
   conda activate nucnet-env
   ```

2. **Date & Biblioteci**:
   - Notebook-ul verifică automat prezența directoarelor `libs/` și `data/`.
   - Dacă acestea nu există, codul le va dezarhiva automat din fișierele `libs.zip` și `data.zip` aflate în directorul curent.

3. **Rularea Notebook-ului**:
   Se pornește Jupyter Notebook sau Jupyter Lab în mediul activat:
   ```bash
   jupyter notebook local_inference_demo.ipynb
   ```
   *Notă: Nu este recomandată rularea acestui notebook pe un procesor (CPU) din cauza cerințelor computaționale extrem de ridicate ale convoluțiilor 3D disperse (sparse 3D convolutions).*

---

## 2. Demo Inferență Google Colab (`colab_inference_demo.ipynb`)

Acest notebook este optimizat pentru a fi rulat pe Google Colab, folosind resursele GPU gratuite oferite în cloud (precum T4).

### Instrucțiuni de Configurare

1. **Deschidere în Colab**:
   Fișierul `colab_inference_demo.ipynb` se încarcă direct pe [Google Colab](https://colab.research.google.com/). Toate dependințele necesare se găsesc în sațiul: https://drive.google.com/drive/folders/1W6LsJhskAkm7kke2cawEFdHPXm0JzNfe?usp=sharing.

2. **Activare GPU**:
   În meniul superior din Colab, se accesează **Runtime > Change runtime type** și se selectează un GPU (ex: T4 GPU).

3. **Rularea Notebook-ului**:
   - Celulele se rulează secvențial.
   - Prima celulă de cod va solicita autentificarea cu un cont Google. Această autentificare este necesară pentru a descărca în siguranță librăriile custom (`libs.zip`) și datele necesare (`data.zip`) direct de pe Google Drive în mediul Colab.
   - Notebook-ul va dezarhiva automat aceste fișiere și va instala orice dependențe PyTorch care lipsesc (cum ar fi `torch-scatter` și `spconv`).
