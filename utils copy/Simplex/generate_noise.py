import torch

def generate_simplex_4noise(
    Simplex_instance,
    x, 
    t, 
    octave=6, 
    persistence=0.8, 
    frequency=64,
    in_channels=4
):
    noise = torch.empty(x.shape).to(x.device)
    for i in range(in_channels):
        noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                    Simplex_instance.rand_4d_fixed_T_octaves(
                        shape = x.shape[2:], 
                        T = t.detach().cpu().numpy(), 
                        scale = 0.1, 
                        octaves = octave,
                        persistence = persistence,
                        frequency = frequency
                    )
                ).to(x.device), 0
        ).repeat(x.shape[0], 1, 1, 1, 1)
    return noise
# shape, T, scale = 0.1, octaves=1, persistence=0.5, frequency=32

def generate_simplex_noise(
    Simplex_instance,
    x, 
    t, 
    random_param=False, 
    octave=6, 
    persistence=0.8, 
    frequency=64,
    in_channels=4
):
    noise = torch.empty(x.shape).to(x.device)
#     print("x", x.shape)
#     print("noise", noise.shape)
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            param = random.choice(
                    [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                     (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                     (2, 0.85, 8),
                     (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                     (1, 0.85, 8),
                     (1, 0.85, 4), (1, 0.85, 2), ]
                    )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                    torch.from_numpy(
                            # Simplex_instance.rand_2d_octaves(
                            #         x.shape[-2:], param[0], param[1],
                            #         param[2]
                            #         )
                            Simplex_instance.rand_3d_fixed_T_octaves(
                                    x.shape[-2:], t.detach().cpu().numpy(), param[0], param[1],
                                    param[2]
                                    )
                            ).to(x.device), 0
                    ).repeat(x.shape[0], 1, 1, 1)
#         print(f"i and in_channels {i} / {in_channels}")
#         print("noise shape", noise.shape)
#         print("t shape", t.shape)
        noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                        # Simplex_instance.rand_2d_octaves(
                        #         x.shape[-2:], octave,
                        #         persistence, frequency
                        #         )
                        Simplex_instance.rand_3d_fixed_T_octaves(
                                x.shape[-2:], t.detach().cpu().numpy(), octave,
                                persistence, frequency
                                )
                        ).to(x.device), 0
                ).repeat(x.shape[0], 1, 1, 1)
    return noise
