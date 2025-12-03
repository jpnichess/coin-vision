import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://coin-vision-31509-default-rtdb.firebaseio.com/"
})

# 2. Função para enviar a moeda
def enviar_moeda(valor):
    ref = db.reference("moeda/valor")
    ref.set(valor)
    print(f"Valor enviado para o Firebase: {valor}")
