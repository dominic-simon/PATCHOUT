import torch 
import torch.nn as nn
import torch.nn.functional as F

class PatchDetector(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(1000, 512)
		self.fc2 = nn.Linear(512, 128)
		self.mp = nn.MaxPool1d(3, stride=2)
		self.fc3 = nn.Linear(63, 32)
		self.fc4 = nn.Linear(32, 2)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.mp(x)
		x = self.fc3(x)
		x = self.relu(x)
		x = self.fc4(x)
		x = self.relu(x)
		return x