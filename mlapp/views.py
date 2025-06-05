from django.shortcuts import render
from .forms import PlayerForm
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from django.core.files.storage import FileSystemStorage

# Load model dan scaler
model = load_model('mlapp/ml/model.h5')
scaler = joblib.load('mlapp/ml/scaler.pkl')

def home(request):
    result = None
    position = None
    rating = None
    photo_url = None
    template_code = 'gold'  # default template

    if request.method == 'POST':
        form = PlayerForm(request.POST, request.FILES)

        if form.is_valid():
            # Ambil data input user
            data = np.array([[  
                form.cleaned_data['physical'],
                form.cleaned_data['speed_agility'],
                form.cleaned_data['attack'],
                form.cleaned_data['pass_control'],
                form.cleaned_data['defense'],
                form.cleaned_data['free_kick'],
                form.cleaned_data['mental']
            ]])

            # Normalisasi dan prediksi kaki dominan
            scaled = scaler.transform(data)
            pred = model.predict(scaled)[0][0]
            result = "Right Foot" if pred >= 0.5 else "Left Foot"

            # Rule-based posisi ideal
            atk = form.cleaned_data['attack']
            dfn = form.cleaned_data['defense']
            spd = form.cleaned_data['speed_agility']
            pas = form.cleaned_data['pass_control']
            men = form.cleaned_data['mental']
            phy = form.cleaned_data['physical']

            if atk > 75 and dfn < 50:
                position = "ST (Striker)"
            elif dfn > 70 and phy > 60:
                position = "CB (Center Back)"
            elif pas + men > 130:
                position = "AM (Attacking Midfielder)"
            elif spd > 75:
                position = "Winger"
            else:
                position = "CM (Central Midfielder)"

            # Hitung rating FIFA-style
            rating_speed = spd
            rating_attack = (atk + form.cleaned_data['free_kick']) / 2
            rating_defense = dfn
            rating_passing = pas
            rating_physical = phy
            rating_overall = np.mean([
                rating_speed, rating_attack, rating_defense,
                rating_passing, rating_physical
            ])

            rating = {
                'speed': int(rating_speed),
                'attack': int(rating_attack),
                'defense': int(rating_defense),
                'passing': int(rating_passing),
                'physical': int(rating_physical),
                'overall': int(rating_overall)
            }

            # Ambil pilihan template user
            template_code = form.cleaned_data['template_choice']

            # Upload dan simpan foto user jika ada
            uploaded_photo = request.FILES.get('photo')
            if uploaded_photo:
                fs = FileSystemStorage()
                filename = fs.save(uploaded_photo.name, uploaded_photo)
                photo_url = fs.url(filename)

    else:
        form = PlayerForm()

    return render(request, 'index.html', {
        'form': form,
        'result': result,
        'position': position,
        'rating': rating,
        'photo_url': photo_url,
        'template_code': template_code,
    })
