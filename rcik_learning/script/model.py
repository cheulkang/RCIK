import torch
import torch.nn as nn


class ManipulationConvNet(nn.Module):
    def __init__(self, dof):
        super(ManipulationConvNet, self).__init__()
        self.dof = dof

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
        )

        self.fc = nn.Sequential(
            nn.Linear((self.dof+2) * 128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ManipulationMLPNet(nn.Module):
    def __init__(self, dof):
        super(ManipulationMLPNet, self).__init__()
        self.dof = dof

        self.fc = nn.Sequential(
            nn.Linear(self.dof, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.fc(x)
        return x


class EnvironmentEncoder(nn.Module):
    def __init__(self):
        super(EnvironmentEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2), # batch, 32, 18, 18, 18
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3), # batch, 32, 16, 16, 16
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # batch, 64, 8, 8, 8
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # batch, 64, 8, 8, 8
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # batch, 128, 4, 4, 4
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # batch, 128, 4, 4, 4
            nn.BatchNorm3d(128),
        )

        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return x


class EnvironmentDecoder(nn.Module):
    def __init__(self):
        super(EnvironmentDecoder, self).__init__()

        self.headTrans = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128*4*4*4),
        )

        self.conv2Trans = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
        )

        self.conv1Trans = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=6, stride=2),
        )

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.headTrans(x)
        x = x.view(x.size(0), 128, 4, 4, 4)
        x = self.conv2Trans(x)
        x = self.conv1Trans(x)
        x = self.sigmoid(x)

        return x


class RCIK(nn.Module):
    def __init__(self, dof):
        super(RCIK, self).__init__()
        self.dof = dof

        self.extractor_conf = ManipulationConvNet(self.dof)
        self.extractor_env = EnvironmentEncoder()

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained(self, checkpoint):
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.extractor_env.state_dict()}
        self.extractor_env.load_state_dict(pretrained_dict)

    def get_env_feature(self, env):
        return self.extractor_env(env)

    def forward(self, conf, env):
        # if self.training is False:
        #     assert env.size(0) == 1, "You should pass only single environment map."
        #     env = env.repeat(conf.size(0), 1, 1, 1, 1)

        f_conf = self.extractor_conf(conf)
        # f_env = self.extractor_env(env)
        # f_env = self.get_env_feature(env)
        f_env = env
        f_env = f_env.view(f_env.size(0), -1)

        f = torch.cat([f_conf, f_env], dim=1)

        out = self.fc(f)

        return out
