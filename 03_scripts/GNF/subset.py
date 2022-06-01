import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import NF
import plotly.express as px
import plotly.graph_objs as go

from utils import *

def define_GNF(device, conditioner_args, A=None):
    # Zones ordering = [1'TOT', 2'CT', 2'ME', 2'NH', 2'RI', 2'VT', 2'MASS', 3'NEMASSBOST', 3'WCMASS', 3'SEMASS']

    # prepare flow
    #conditioner_args['l1'] = 1
    # parameter linked to continuous gumble law (approximation of maximum in a random vector)
    conditioner_args['gumble_T'] = 0.5

    conditioner_args['nb_epoch_update'] = 2

    conditioner_args["hot_encoding"] = True
    if A is not None:
        conditioner_args["A_prior"] = A

    conditioner = NF.DAGConditioner(**conditioner_args)
    normalizer = NF.AffineNormalizer()

    flow_steps = [NF.NormalizingFlowStep(conditioner, normalizer)]
    flow = NF.FCNormalizingFlow(flow_steps, NF.NormalLogDensity())
    flow.to(device)

    return flow

def train_GNF(flow, train_set, val_set, nb_epoch, title_loss_plot, device):
    opt = torch.optim.Adam(flow.parameters(), 1e-4, weight_decay=1e-5)

    print('Start training GNF.. \n')
    train_loss = []
    val_loss = []

    for epoch in range(nb_epoch):
        loss_tot = 0

        for X in train_set:

            cur_x = torch.tensor(X, dtype=torch.float).to(device)
            #print(cur_x)
            z, jac = flow(cur_x)
            loss = flow.loss(z, jac)
            loss_tot += loss.detach()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        flow.step(epoch, loss_tot)
        #A_debugger = flow.getConditioners()[0].A
        #print(A_debugger)

        if epoch % 5 == 0:
            mean_train_loss = loss_tot / (train_set.shape[0])
            train_loss.append(mean_train_loss.cpu())
            print("Epoch {} Mean Loss: {:3f}".format(epoch, mean_train_loss))

            mean_val_loss = 0
            for X in val_set:
                cur_x = torch.tensor(X, dtype=torch.float).to(device)
                z, jac = flow(cur_x)
                loss = flow.loss(z, jac)
                mean_val_loss += loss.detach().cpu()

            val_loss.append(mean_val_loss / val_set.shape[0])

    print("\nGNF training finished!\n")

    fig = go.Figure()
    abs = np.arange(0, nb_epoch, 5)
    fig.add_trace(go.Scatter(x=abs, y=train_loss, name="training loss"))
    fig.add_trace(go.Scatter(x=abs, y=val_loss, name="validation loss"))
    fig.update_layout(
        title="Losses during training of GNF l1={}".format(title_loss_plot),
        xaxis_title="Epochs",
        yaxis_title="Negative log likelihood"
    )
    fig.update_yaxes(
        range=[0, 350]
    )

    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/l1{}_GNF_TrainingLoss.png".format(title_loss_plot))

    return flow

dataraw_path = "../../02_datasets/Raw/gefcom2017-d.csv"
# dr -> dataraw
dr = pd.read_csv(dataraw_path)

dr = dr[dr.zone.isin(["SEMASS", "WCMASS", "NEMASSBOST"])][dr.ts > '2014-03-01 00:00:00'][dr.ts < '2016-03-01 00:00:00']
grp_date = dr.groupby(by=["date"])

zone_ordering = ['NEMASSBOST', 'WCMASS', 'SEMASS']
acc_load = []
acc_context = []

for date, frame in grp_date:
    if len(frame) != (3*24):
        pass
    else:
        frame['zone'] = pd.Categorical(frame['zone'], zone_ordering)
        frame = frame.sort_values(["zone", "hour"])

        acc_load.append(frame["demand"].to_numpy())

day_load = np.vstack(acc_load)

train_set, val_set = train_test_split(day_load, test_size=0.2,
                                            train_size=0.8, shuffle=False)

val_set = val_set[:int(val_set.shape[0] / 10) * 10]
train_set = train_set[:int(train_set.shape[0] / 10) * 10]
BATCH_SIZE = 10

train_std_target = StandardScaler()
train_set = train_std_target.fit_transform(train_set)
train_set = train_set.reshape((-1, BATCH_SIZE, 3*24))

val_std_target = StandardScaler()
val_set = val_std_target.fit_transform(val_set)
val_set = val_set.reshape((-1, BATCH_SIZE, 3*24))

device = None
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('device used: ', device, "\n")

conditioner_args = {}
# 10 [zones] * 24 [hours]
conditioner_args["in_size"] = 3*24
# conditioner_args["cond_in"] = CONTEXT_SIZE
conditioner_args["hidden"] = [200, 100, 50]
conditioner_args["out_size"] = 2

N_EPOCHS = 100
l1_list = [0.5, 1, 2, 5, 10]


for l1 in l1_list:
    # run training
    # define experiment
    A = torch.Tensor(3*24, 3*24).to(device)
    A = A.fill_(0.5)
    conditioner_args['l1'] = l1

    GNF_model = define_GNF(device, conditioner_args, A)

    title_loss_plot = l1
    GNF_model = train_GNF(GNF_model, train_set, val_set, N_EPOCHS, title_loss_plot, device)

    n_scen = 5
    GNF_scen_sample = []
    for i in range(n_scen):
        z = torch.randn(3*24).unsqueeze(0).to(device)
        scen_sample = GNF_model.invert(z).detach().cpu()
        scen_sample = train_std_target.inverse_transform(scen_sample)
        GNF_scen_sample.append(scen_sample)
    print(GNF_scen_sample)
    colors = px.colors.qualitative.Light24[:10]
    fig = go.Figure()
    for scen in GNF_scen_sample:
        scen = scen.reshape((3, 24))
        # sum bottom nodes
        scen = np.vstack((scen, np.sum(scen, axis=0)))

        for zone, col in zip(scen, colors):
            fig.add_trace(go.Scatter(x=np.arange(24), y=zone, line=dict(color=col)))
    fig.update_layout(
        title="Scenario of day",
        xaxis_title="Hour",
        yaxis_title="",
        showlegend=False,
        font=dict(
            size=20
        )
    )
    fig.write_image("images/{}_scenarios.png".format(l1))