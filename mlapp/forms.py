from django import forms

TEMPLATE_CHOICES = [
    ('gold', 'Gold Rare'),
    ('toty', 'TOTY'),
    ('silver', 'Silver'),
    ('bronze', 'Bronze'),
]

class PlayerForm(forms.Form):
    physical = forms.IntegerField(min_value=0, max_value=100, label="Physical")
    speed_agility = forms.IntegerField(min_value=0, max_value=100, label="Speed & Agility")
    attack = forms.IntegerField(min_value=0, max_value=100, label="Attack")
    pass_control = forms.IntegerField(min_value=0, max_value=100, label="Pass & Control")
    defense = forms.IntegerField(min_value=0, max_value=100, label="Defense")
    free_kick = forms.IntegerField(min_value=0, max_value=100, label="Free Kick")
    mental = forms.IntegerField(min_value=0, max_value=100, label="Mental")
    
    template_choice = forms.ChoiceField(
        choices=TEMPLATE_CHOICES,
        label="Pilih Template FIFA Card",
        widget=forms.Select()
    )

    photo = forms.ImageField(label="Upload Foto", required=False)
