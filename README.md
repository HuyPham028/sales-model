### If you are using an editor (e.g., VS Code)

#### 1. Create a virtual environment
```bash
python -m venv venv
```

#### 2. Activate the environment  
If you are on **Windows (PowerShell)**:
```bash
.\venv\Scripts\Activate.ps1
```

If you are on **Linux/macOS**:
```bash
source venv/bin/activate
```

#### 3. Install necessary packages
```bash
pip install -r requirements.txt
```

#### 4. Select the virtual environment kernel  
In your editor (e.g., VS Code or Jupyter), change the kernel to the newly created **venv**, then run all cells and explore.

---

### Alternatively: Google Colab  
You can open the notebook directly in Google Colab and run it without installing anything locally.

---

### Running the Streamlit app  
Navigate to the `streamlit` folder and run:
```bash
streamlit run app.py
```
