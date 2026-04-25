from torch_geometric.loader import DataLoader

from ApexDAG.nn.models.v2.gat import MultiTaskGATv2
from ApexDAG.nn.models.v2.loss import MultiTaskUncertaintyLoss
from ApexDAG.nn.trainer_v2 import ApexOnlineTrainer

graph_files = []

train_dataset = [torch.load(f) for f in graph_files]
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = MultiTaskGATv2()
criterion = MultiTaskUncertaintyLoss()
trainer = ApexOnlineTrainer(model, criterion)

trainer.log_model_graph(train_dataset[0])

for epoch in range(50):
    for batch in train_loader:
        loss = trainer.train_step(batch)
    print(f"Epoch {epoch} complete.")

trainer.save_cpu_checkpoint("apexdag_v2_cpu.pt")


"""
Online Training Example:
new_annotated_data = self.process_json_to_pyg(self.get_json_body())
self.graph_buffer.append(new_annotated_data)

if len(self.graph_buffer) >= self.update_threshold:
    batch = next(iter(DataLoader(self.graph_buffer, batch_size=self.update_threshold)))
    self.trainer.train_step(batch) 
    self.graph_buffer.clear()
    self.trainer.save_cpu_checkpoint("apexdag_v2_cpu_latest.pt")
"""
