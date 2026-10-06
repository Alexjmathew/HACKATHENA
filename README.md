# GST Fraud Detection - Neural Network + Flask

    pip install -r requirements.txt
    python generate_dataset.py     # 1. creates data/train_dataset.csv and data/transactions.csv
    python train_model.py          # 2. trains the neural network -> model/
    python app.py                  # 3. open http://127.0.0.1:5000 and upload data/transactions.csv

Your own CSV needs: seller_gstin, buyer_gstin, amount.
Optional: invoice_id, tax_rate, itc_amount, seller_age_days, invoice_date.
data/transactions_with_answers.csv is the same test file WITH labels, for checking accuracy.
