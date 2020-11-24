import torch
import numpy as np
print(torch.__version__)
print(torch.cuda.is_available())

a = np.array([1, 2, 3])
t = torch.from_numpy(a)
bolb = t.cuda().unsqueeze(0)
print(t)
print(bolb)

tracked_stracks = []
tracked_stracks.append([1,2])
tracked_stracks.append([3,4])

# # 将list形式转换为array形式
# multi_mean = np.asarray([st.mean.copy() for st in tracked_stracks])
# print(multi_mean)

for track in tracked_stracks:
    print(track)

aa = 0
def tets(aa):
    aa = 10
    print(aa)
tets(aa)
print(aa)

def joint_stracks(tlista):
    res = []
    for t in tlista:
        res.append(t)
    return res
pool = joint_stracks(tracked_stracks)

print(pool)


