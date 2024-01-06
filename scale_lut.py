def scale_adaptation(mod, FE, FR):
    if mod:
        if 0 <= FE <= 1:
            if FR == 0.1:
                scale = 1.0
            elif 0.2 <= FR <= 0.5:
                scale = 0.2
            else:
                scale = 0.1

        elif 2 <= FE <= 4:
            if FR == 0.1:
                scale = 1.0
            elif 0.2 <= FR <= 0.5:
                scale = 0.2
            else:
                scale = 0.1

        else:
            if FR == 0.1:
                scale = 0.8
            elif 0.2 <= FR <= 0.5:
                scale = 0.2
            else:
                scale = 0.1

    else:
        if FE == 0:
            if 0.1 <= FR <= 0.2:
                scale = 1.0
            elif 0.3 <= FR <= 0.6:
                scale = 0.2
            else:
                scale = 0.1

        elif FE == 1:
            if FR == 0.1:
                scale = 1.0
            elif FR == 0.2:
                scale = 0.8
            elif 0.3 <= FR <= 0.6:
                scale = 0.4
            else:
                scale = 0.2

        elif 2 <= FE <= 4:
            if FR == 0.1:
                scale = 1.0
            elif FR == 0.2:
                scale = 0.6
            elif 0.3 <= FR <= 0.6:
                scale = 0.4
            else:
                scale = 0.2

        else:
            if FR == 0.1:
                scale = 0.8
            elif FR == 0.2:
                scale = 0.6
            elif 0.3 <= FR <= 0.6:
                scale = 0.4
            else:
                scale = 0.2

    return scale
