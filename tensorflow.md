# TensorFlow installation in anaconda steps:
```console
conda create -n tf_env python=3.10
conda activate tf_env
pip install ipykernel
pip install tensorflow
python -m ipykernel install --user --name=tfkernel
```

---

Perfect ✅ — since you're using **Anaconda Terminal**, this is actually *much easier* and better for ML.

You should **NOT use `venv`**.
With Anaconda, you should use **conda environments instead**.

---

# ✅ Best Way to Install TensorFlow in Anaconda

Open **Anaconda Prompt**, then:

---

## 1️⃣ Create a New Environment

I recommend Python 3.10 (most stable with TensorFlow):

```bash
conda create -n tf_env python=3.10
```

Press `y` when asked.

---

## 2️⃣ Activate It

```bash
conda activate tf_env
```

You should now see:

```
(tf_env) C:\Users\...
```

---

## 3️⃣ Install TensorFlow

Now install TensorFlow using pip (recommended way inside conda):

```bash
pip install tensorflow
```

Do NOT use:

```
conda install tensorflow
```

It often causes dependency issues.

---

## 4️⃣ Add Environment to Jupyter

If you use Jupyter Notebook:

```bash
pip install ipykernel
python -m ipykernel install --user --name=tf_env --display-name "Python (tf_env)"
```

Now open Jupyter and select:

👉 **Kernel → Change Kernel → Python (tf_env)**

---

# 🎯 Why This Is the Best Setup

* No global install
* No Windows PATH issues
* No Microsoft Store problems
* Clean ML environment
* Easy to delete if something breaks

---

# 🔎 Quick Test

After installation, run:

```python
import tensorflow as tf
print(tf.__version__)
```

If it prints a version number → you're done 🎉

---

If you want, tell me:

* Are you using Jupyter Notebook or JupyterLab?
* CPU only or do you have NVIDIA GPU?

I’ll optimize the setup for your case.




