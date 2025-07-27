CAPSTONE PROJECT <br>
<br>
<br>
Objective:
Allow doctors to write prescriptions on paper as usual. Then, automatically convert scanned handwritten prescriptions into structured, digital text using AI — correcting handwriting mistakes with medical vocabulary.<br>
<br>
Key Features of the Handwritten Medical OCR System<br>
<br>
1.Scanned Input Support: Accepts handwritten prescriptions as scanned images or photos.<br>
2.Image Preprocessing: Enhances image quality using noise removal, binarization, and alignment.<br>
3.Region Segmentation: Detects and focuses only on handwritten parts of the prescription.<br>
4.Handwritten Text Recognition: Uses TrOCR or ViT to convert handwriting to digital text.<br>
5.Medical Spell Correction: Fixes recognition errors using a medical vocabulary and language models.<br>
6.Structured Output: Extracts and organizes data like patient name, symptoms, medicines, and advice.<br>
7.Secure & Offline: Can run offline without compromising patient privacy.<br>
8.Affordable: Software-based solution with minimal hardware requirements.<br>
<br>
<hr style = "height:0.5px;">
Tech Stack<br>
🖼 Image Processing<br>
OpenCV, Pillow – for preprocessing (noise removal, binarization, skew correction)<br>
<br>
🧠 OCR & AI Models<br>
TrOCR (HuggingFace Transformers) – for handwriting recognition<br>
Vision Transformer (ViT) – for image classification or segmentation (optional)<br>
PyTorch / TensorFlow – deep learning backend<br>
<br>
📊 Data Annotation & Management<br>
Label Studio, CVAT – for creating labeled datasets<br>
Pandas, JSON/CSV – for handling structured data<br>
🧠 Error Correction<br>
SymSpell, FuzzyWuzzy, SpaCy – for spelling correction using medical vocabulary<br>
Custom language model – for context-aware correction<br>
<br>
📄 Output Generation<br>
python-docx, ReportLab – to generate .docx or PDF with corrected prescription<br>
Streamlit / Flask – for a simple user interface <br>
<br>
<hr style = "height:0.5px;">
USE CASES:<br>
🏥Digitize handwritten prescriptions in hospitals and clinics<br>
💊 Assist pharmacies in accurate medicine dispensing<br>
📁 Integrate with EHRs for better patient record management<br>
🌍 Support rural healthcare with offline-friendly digitization<br>
📊 Enable medical data analytics and drug usage tracking<br>
🧾 Simplify insurance claims with clean prescription data<br>
