import os
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model_Unet import UNet
from data_prep import get_loaders

class DiceLoss(nn.Module):
    """
    Klasa implementująca funkcję straty Dice (Dice Loss).
    Mierzy stopień nakładania się masek predykcyjnych i rzeczywistych,
    co jest kluczowe przy silnym niezrównoważeniu klas.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice


def find_best_checkpoint(checkpoint_dir="checkpoints"):
    """Automatycznie wyszukuje plik wag z najniższą wartością funkcji straty."""
    search_path = os.path.join(checkpoint_dir, "*.pth")
    checkpoints = glob.glob(search_path)
    
    if not checkpoints:
        return None
        
    #Sortowanie plików według wartości straty wyekstrahowanej z nazwy pliku
    try:
        best_checkpoint = min(checkpoints, key=lambda x: float(x.split("loss_")[-1].replace(".pth", "")))
        return best_checkpoint
    except ValueError:
        return checkpoints[-1] #Koło ratunkowe - zwraca ostatnio zmodyfikowany plik


def evaluate_and_visualize(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Funkcja przeprowadzająca ewaluację końcową modelu na wydzielonym zbiorze testowym.
    Oblicza metryki (BCE, Dice) oraz generuje wykresy i wizualizacje segmentacji.
    """
    print(f"Uruchamianie ewaluacji na urządzeniu: {device}")
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    #Automatyczne wykrywanie i ładowanie najlepszych wag
    checkpoint_path = find_best_checkpoint()
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Pomyślnie załadowano najlepsze wyuczone wagi: {checkpoint_path}")
    else:
        print("⚠ Nie znaleziono plików wag w katalogu 'checkpoints'.")
        print("Uruchamianie testu w trybie demonstracyjnym na losowo zainicjalizowanych wagach.")
    
    model.eval()
    
    #Pobranie loaderów (interesuje nas wyłącznie niezależny test_loader)
    _, _, test_loader = get_loaders(batch_size=4, test_size=0.5, random_state=42)
    
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()
    
    total_bce, total_dice, total_loss = 0.0, 0.0, 0.0
    bce_history, dice_history, total_history = [], [], []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            #Dopasowanie wymiaru maski z [B, H, W] do [B, 1, H, W] na potrzeby funkcji straty
            masks = masks.to(device).unsqueeze(1)
            
            outputs = model(images)
            
            loss_bce = bce_criterion(outputs, masks)
            loss_dice = dice_criterion(outputs, masks)
            loss_combined = loss_bce + loss_dice
            
            total_bce += loss_bce.item()
            total_dice += loss_dice.item()
            total_loss += loss_combined.item()
            
            bce_history.append(loss_bce.item())
            dice_history.append(loss_dice.item())
            total_history.append(loss_combined.item())
            
            #Generowanie wizualizacji dla pierwszej paczki danych (Batch #0)
            if batch_idx == 0:
                visualize_predictions(images, masks, outputs)
                
    num_batches = len(test_loader)
    print("\n" + "="*50)
    print("=== PODSUMOWANIE METRYK (ZBIÓR TESTOWY) ===")
    print(f"Średnia hybrydowa funkcja straty (Total Hybrid Loss): {total_loss / num_batches:.4f}")
    print(f"Średnia binarna entropia krzyżowa (BCE Loss): {total_bce / num_batches:.4f}")
    print(f"Średnia strata Dice'a (Dice Loss): {total_dice / num_batches:.4f}")
    print("="*50)
    
    plot_loss_curves(bce_history, dice_history, total_history)


def visualize_predictions(images, masks, outputs, num_samples=3):
    """Generuje i zapisuje na dysku porównanie masek: Oryginał -> Ground Truth -> U-Net."""
    images = images.cpu()
    masks = masks.cpu()
    preds = torch.sigmoid(outputs).cpu() > 0.5
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    
    axes[0, 0].set_title("Obraz oryginalny (SAR)", fontsize=12)
    axes[0, 1].set_title("Maska rzeczywista (Ground Truth)", fontsize=12)
    axes[0, 2].set_title("Predykcja modelu (U-Net)", fontsize=12)
    
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        true_mask = masks[i, 0].numpy()
        pred_mask = preds[i, 0].numpy()
        
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 2].imshow(pred_mask, cmap='gray')
        
        for ax in axes[i]:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("wykresy_wyniki_segmentacji.png", dpi=300)
    print("✓ Zapisano wykres wizualizacji: wykresy_wyniki_segmentacji.png")
    plt.show()


def plot_loss_curves(bce, dice, total):
    """Tworzy wykres liniowy prezentujący zachowanie funkcji strat na zbiorze testowym."""
    plt.figure(figsize=(10, 5))
    plt.plot(total, label='Total Hybrid Loss (BCE + Dice)', color='purple', linewidth=2)
    plt.plot(bce, label='BCE Loss', color='blue', linestyle='--')
    plt.plot(dice, label='Dice Loss', color='orange', linestyle='--')
    plt.title('Analiza przebiegu funkcji strat na zbiorze testowym (per Batch)')
    plt.xlabel('Indeks paczki danych (Batch)')
    plt.ylabel('Wartość błędu')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig("wykres_analiza_strat.png", dpi=300)
    print("✓ Zapisano wykres analizy strat: wykres_analiza_strat.png")
    plt.show()


if __name__ == "__main__":
    evaluate_and_visualize()