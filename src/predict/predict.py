import torch
import torch.nn.functional as F

# inference of one sequence
def predict(model, sequence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(sequence)
        probabilities = F.softmax(output, dim=1)
        probabilities_list = probabilities.squeeze().tolist()
    return probabilities_list
