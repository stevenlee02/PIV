PIV Praktikum Informationsvisualisierung LMU 2526ws

To Launch:

conda activate piv_env

cd -> text-vis -> frontend
npx serve -p 8000

cd -> text-vis -> backend
uvicorn app:app --reload --port 8001
