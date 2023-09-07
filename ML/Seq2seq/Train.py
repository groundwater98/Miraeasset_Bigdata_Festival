import torch


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device) # src, trg => GPU
        optimizer.zero_grad() # gradient 초기화
        output = model(src, trg)

        output_dim = output.shape[-1] # 출력 차원 가져오기
        output = output[1:].view(-1, output_dim) # 첫 번째 토큰인 <sos> 제외
        trg = trg[1:].view(-1)
        loss = criterion(output, trg) # 예측된 output과 실제 trg를 사용하여 loss 계산
        loss.backward() # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # gradient 폭발을 방지하기 위해 gradient 크기를 제한
        optimizer.step() # 계산된 gradient를 바탕으로 model parameter update
        epoch_loss += loss.item() 

    return epoch_loss / len(iterator) # average loss 반환


def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)