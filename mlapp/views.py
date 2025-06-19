from django.shortcuts import render
from .forms import PlayerForm
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from django.core.files.storage import FileSystemStorage
import json
import os

# Load model dan scaler
model = load_model('mlapp/ml/model.h5')
scaler = joblib.load('mlapp/ml/scaler.pkl')

def home(request):
    result = None
    position = None
    rating = None
    photo_url = None
    template_code = 'gold'  # default template
    player_name = None
    position_scores = None
    alternative_positions = None
    similar_players = None
    transfer_value = None
    user_badge = None

    if request.method == 'POST':
        form = PlayerForm(request.POST, request.FILES)

        if form.is_valid():
            # Ambil data input user
            player_name = form.cleaned_data['name']
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

            # Extract all attributes for position calculation
            atk = form.cleaned_data['attack']
            dfn = form.cleaned_data['defense']
            spd = form.cleaned_data['speed_agility']
            pas = form.cleaned_data['pass_control']
            men = form.cleaned_data['mental']
            phy = form.cleaned_data['physical']
            fk = form.cleaned_data['free_kick']

            # Advanced Position Scoring System
            position_scores = {
                "ST (Striker)": {
                    "score": (atk * 0.4 + spd * 0.25 + phy * 0.15 + men * 0.1 + pas * 0.1),
                    "requirements": {
                        "attack": atk >= 70,
                        "speed": spd >= 65,
                        "physical": phy >= 60
                    }
                },
                "CF (Center Forward)": {
                    "score": (atk * 0.35 + pas * 0.25 + men * 0.2 + spd * 0.1 + phy * 0.1),
                    "requirements": {
                        "attack": atk >= 65,
                        "passing": pas >= 70,
                        "mental": men >= 65
                    }
                },
                "WF (Wing Forward)": {
                    "score": (spd * 0.35 + atk * 0.25 + pas * 0.2 + men * 0.1 + fk * 0.1),
                    "requirements": {
                        "speed": spd >= 75,
                        "attack": atk >= 65,
                        "passing": pas >= 65
                    }
                },
                "CAM (Attacking Midfielder)": {
                    "score": (pas * 0.35 + men * 0.25 + atk * 0.2 + fk * 0.1 + spd * 0.1),
                    "requirements": {
                        "passing": pas >= 75,
                        "mental": men >= 70,
                        "attack": atk >= 60
                    }
                },
                "CM (Central Midfielder)": {
                    "score": (pas * 0.3 + men * 0.25 + phy * 0.15 + dfn * 0.15 + atk * 0.15),
                    "requirements": {
                        "passing": pas >= 70,
                        "mental": men >= 70,
                        "physical": phy >= 65
                    }
                },
                "CDM (Defensive Midfielder)": {
                    "score": (dfn * 0.35 + pas * 0.25 + phy * 0.2 + men * 0.1 + spd * 0.1),
                    "requirements": {
                        "defense": dfn >= 70,
                        "passing": pas >= 65,
                        "physical": phy >= 70
                    }
                },
                "WB (Wing Back)": {
                    "score": (spd * 0.3 + dfn * 0.25 + pas * 0.2 + phy * 0.15 + men * 0.1),
                    "requirements": {
                        "speed": spd >= 75,
                        "defense": dfn >= 65,
                        "passing": pas >= 65
                    }
                },
                "CB (Center Back)": {
                    "score": (dfn * 0.4 + phy * 0.3 + men * 0.15 + spd * 0.1 + pas * 0.05),
                    "requirements": {
                        "defense": dfn >= 75,
                        "physical": phy >= 70,
                        "mental": men >= 65
                    }
                }
            }

            # Calculate final scores and check requirements
            for pos, data in position_scores.items():
                # Calculate weighted score
                weighted_score = data["score"]
                # Check if meets minimum requirements
                meets_requirements = all(data["requirements"].values())
                # Apply penalty if requirements not met
                if not meets_requirements:
                    weighted_score *= 0.7
                # Update score
                position_scores[pos]["final_score"] = round(weighted_score, 2)
                position_scores[pos]["meets_requirements"] = meets_requirements

            # Sort positions by score
            sorted_positions = sorted(
                position_scores.items(),
                key=lambda x: x[1]["final_score"],
                reverse=True
            )

            # Select primary position and alternatives
            position = sorted_positions[0][0]
            alternative_positions = [
                {
                    "name": pos[0],
                    "score": pos[1]["final_score"],
                    "meets_requirements": pos[1]["meets_requirements"]
                }
                for pos in sorted_positions[1:4]  # Get next 3 best positions
            ]

            # Hitung rating FIFA-style dengan formula yang lebih kompleks
            rating_speed = spd
            rating_attack = (atk * 0.7 + fk * 0.3)
            rating_defense = (dfn * 0.8 + phy * 0.2)
            rating_passing = (pas * 0.7 + men * 0.3)
            rating_physical = (phy * 0.7 + men * 0.3)
            
            # Adjust overall rating based on position
            position_weights = {
                "ST (Striker)": {"attack": 0.4, "speed": 0.3, "passing": 0.15, "physical": 0.1, "defense": 0.05},
                "CF (Center Forward)": {"attack": 0.35, "passing": 0.25, "speed": 0.2, "physical": 0.15, "defense": 0.05},
                "WF (Wing Forward)": {"speed": 0.35, "attack": 0.25, "passing": 0.25, "physical": 0.1, "defense": 0.05},
                "CAM (Attacking Midfielder)": {"passing": 0.35, "attack": 0.25, "speed": 0.2, "physical": 0.1, "defense": 0.1},
                "CM (Central Midfielder)": {"passing": 0.3, "physical": 0.2, "defense": 0.2, "attack": 0.15, "speed": 0.15},
                "CDM (Defensive Midfielder)": {"defense": 0.35, "passing": 0.25, "physical": 0.2, "speed": 0.1, "attack": 0.1},
                "WB (Wing Back)": {"speed": 0.3, "defense": 0.25, "passing": 0.2, "physical": 0.15, "attack": 0.1},
                "CB (Center Back)": {"defense": 0.4, "physical": 0.3, "speed": 0.15, "passing": 0.1, "attack": 0.05}
            }

            weights = position_weights[position]
            rating_overall = (
                rating_attack * weights["attack"] +
                rating_speed * weights["speed"] +
                rating_defense * weights["defense"] +
                rating_passing * weights["passing"] +
                rating_physical * weights["physical"]
            )

            rating = {
                'speed': int(rating_speed),
                'attack': int(rating_attack),
                'defense': int(rating_defense),
                'passing': int(rating_passing),
                'physical': int(rating_physical),
                'overall': int(rating_overall)
            }

            # --- Cari pemain nyata paling mirip ---
            # Path ke file JSON
            real_players_path = os.path.join(os.path.dirname(__file__), 'ml', 'real_players.json')
            with open(real_players_path, 'r', encoding='utf-8') as f:
                real_players = json.load(f)

            # Ambil stats user
            user_stats = {
                'speed': rating['speed'],
                'attack': rating['attack'],
                'defense': rating['defense'],
                'passing': rating['passing'],
                'physical': rating['physical'],
                'overall': rating['overall'],
                'position': position.split('(')[0].strip()  # Ambil kode posisi saja
            }

            # Definisikan bobot perbandingan berdasarkan posisi
            position_weights = {
                'ST': {'speed': 0.25, 'attack': 0.35, 'defense': 0.05, 'passing': 0.20, 'physical': 0.15},
                'CF': {'speed': 0.20, 'attack': 0.30, 'defense': 0.05, 'passing': 0.30, 'physical': 0.15},
                'WF': {'speed': 0.30, 'attack': 0.25, 'defense': 0.05, 'passing': 0.25, 'physical': 0.15},
                'CAM': {'speed': 0.15, 'attack': 0.25, 'defense': 0.10, 'passing': 0.35, 'physical': 0.15},
                'CM': {'speed': 0.15, 'attack': 0.15, 'defense': 0.20, 'passing': 0.30, 'physical': 0.20},
                'CDM': {'speed': 0.10, 'attack': 0.10, 'defense': 0.35, 'passing': 0.25, 'physical': 0.20},
                'WB': {'speed': 0.25, 'attack': 0.10, 'defense': 0.25, 'passing': 0.20, 'physical': 0.20},
                'CB': {'speed': 0.15, 'attack': 0.05, 'defense': 0.40, 'passing': 0.15, 'physical': 0.25}
            }

            # Fungsi untuk mendapatkan bobot berdasarkan posisi
            def get_position_weights(pos):
                # Map posisi yang mirip
                pos_mapping = {
                    'RW': 'WF', 'LW': 'WF',  # Wing Forward
                    'RM': 'WF', 'LM': 'WF',
                    'RB': 'WB', 'LB': 'WB',  # Wing Back
                }
                pos = pos_mapping.get(pos, pos)
                return position_weights.get(pos, position_weights['CM'])  # Default ke CM jika tidak ditemukan

            # Hitung jarak Euclidean dengan bobot
            def weighted_euclidean(user, player):
                weights = get_position_weights(user['position'])
                
                # Hitung similarity posisi (1.0 jika sama, 0.8 jika beda)
                pos_similarity = 1.0 if user['position'] == player['position'] else 0.8
                
                # Hitung jarak dengan bobot untuk setiap atribut
                weighted_sum = sum(
                    weights[attr] * ((float(user[attr]) - float(player[attr])) ** 2)
                    for attr in ['speed', 'attack', 'defense', 'passing', 'physical']
                )
                
                # Tambahkan pengaruh overall rating (dengan bobot lebih kecil)
                overall_diff = (float(user['overall']) - float(player['overall'])) ** 2
                weighted_sum += 0.1 * overall_diff
                
                # Kalikan dengan similarity posisi
                return (weighted_sum ** 0.5) * (1 / pos_similarity)

            # Proses setiap pemain
            for p in real_players:
                p_stats = {
                    'speed': p['speed'],
                    'attack': p['attack'],
                    'defense': p['defense'],
                    'passing': p['passing'],
                    'physical': p['physical'],
                    'overall': p['overall'],
                    'position': p['position']
                }
                p['distance'] = weighted_euclidean(user_stats, p_stats)

            # Urutkan berdasarkan jarak dan ambil 3 terdekat
            similar_players = sorted(real_players, key=lambda x: x['distance'])[:3]

            # Tambahkan similarity score (dalam persen) untuk setiap pemain mirip
            max_possible_distance = 100  # Jarak maksimum yang mungkin
            for p in similar_players:
                similarity_score = max(0, 100 - (p['distance'] * 5))  # Konversi jarak ke similarity score
                p['similarity'] = round(similarity_score, 1)

            # --- END pemain mirip ---

            # Ambil pilihan template user
            template_code = form.cleaned_data['template_choice']

            # Upload dan simpan foto user jika ada
            uploaded_photo = request.FILES.get('photo')
            if uploaded_photo:
                fs = FileSystemStorage()
                filename = fs.save(uploaded_photo.name, uploaded_photo)
                photo_url = fs.url(filename)

            # --- Estimasi Transfer Value ---
            base_value = 1_000_000  # 1 juta euro
            overall = rating['overall']
            # Posisi premium (ST, CAM, RW, LW, CF) lebih mahal
            premium_positions = ['ST', 'CF', 'CAM', 'RW', 'LW']
            pos_code = position.split('(')[0].strip()
            multiplier = 1.0
            if any(p in pos_code for p in premium_positions):
                multiplier = 2.0
            elif 'CB' in pos_code or 'CDM' in pos_code:
                multiplier = 1.3
            elif 'RB' in pos_code or 'LB' in pos_code or 'WB' in pos_code:
                multiplier = 1.1
            # Usia tidak diketahui, jadi tidak dipakai
            transfer_value = int(base_value * (overall/80) ** 2 * multiplier)

            # --- Badge/Medali Unik ---
            stat_map = {
                'speed': 'Speedster',
                'attack': 'Finisher',
                'defense': 'Wall',
                'passing': 'Playmaker',
                'physical': 'Tank',
                'overall': 'Maestro'
            }
            max_stat = max(['speed','attack','defense','passing','physical'], key=lambda s: rating[s])
            user_badge = stat_map[max_stat]

    else:
        form = PlayerForm()

    return render(request, 'index.html', {
        'form': form,
        'result': result,
        'position': position,
        'rating': rating,
        'photo_url': photo_url,
        'template_code': template_code,
        'player_name': player_name,
        'position_scores': position_scores,
        'alternative_positions': alternative_positions,
        'similar_players': similar_players,
        'transfer_value': transfer_value,
        'user_badge': user_badge,
    })
