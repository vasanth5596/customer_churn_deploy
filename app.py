from flask import Flask,render_template,request
import pandas as pd
import pickle

app=Flask(__name__,template_folder='template')
df=pd.read_csv("customer_churn_dataset.csv")
pipe=pickle.load(open('knnmodel.pkl','rb'))

@app.route("/")
def index():
    used_discount=sorted(df["used_discount"].unique())
    used_bogo=sorted(df["used_bogo"].unique())
    zip_code=sorted(df["zip_code"].unique())
    is_referral=sorted(df["is_referral"].unique())
    channel=sorted(df["channel"].unique())
    offer=sorted(df["offer"].unique())
    return render_template('index.html',zip_code=zip_code,
                           channel=channel,
                           offer=offer,
                           used_discount=used_discount,
                           used_bogo=used_bogo,
                           is_referral=is_referral
                           )


@app.route('/predict',methods=['POST'])

def predict():
    recency=request.form.get('recency')
    history=request.form.get('history')
    used_discount=request.form.get('used_discount')
    used_bogo=request.form.get('used_bogo')
    zip_code=request.form.get('zip_code')
    is_referral=request.form.get('is_referral')
    channel=request.form.get('channel')
    offer=request.form.get('offer')

    print(recency,history,used_discount,used_bogo,zip_code,is_referral,channel,offer)
    input=pd.DataFrame([[recency,history,used_discount,used_bogo,zip_code,is_referral,channel,offer]],
                       columns=['recency','history','used_discount','used_bogo','zip_code','is_referral','channel','offer'])

    prediction = pipe.predict(input)

    return str(prediction)



# if __name__=="__main__":
#     app.run(debug=True)
