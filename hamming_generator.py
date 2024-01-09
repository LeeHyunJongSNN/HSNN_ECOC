# GENERATE Q-ARY HAMMING CODE
import numpy as np

def convert(n, m):
    rev_base = ''

    while n > 0:
        n, mod = divmod(n, m)
        rev_base += str(mod)

    return rev_base[::-1]

hamm_n = 11
hamm_k = 8
q = 3
converted = []

for target in range(0, q**hamm_k):
    if len(convert(target, q)) < hamm_k:
        converted.append(list(convert(target, q).zfill(hamm_k)))
    else:
        converted.append(list(convert(target, q)))

hamming_codes = []

for d in converted:
    data = d.copy()
    data.reverse()
    c, ch, j, r, h = 0, 0, 0, 0, []

    while (len(d) + r + 1) > (pow(q, r)):
        r = r + 1

    for i in range(0, (r + len(data))):
        p = q ** c

        if p == (i + 1):
            h.append(0)
            c = c + 1

        else:
            h.append(int(data[j]))
            j = j + 1

    for parity in range(0, (len(h))):
        ph = q ** ch
        if ph == parity + 1:
            startIndex = ph - 1
            i = startIndex
            toXor = []

            while i < len(h):
                block = h[i:i + ph]
                toXor.extend(block)
                i += q * ph

            # for z in range(1, len(toXor)):
            #     h[startIndex] = h[startIndex] ^ toXor[z]
            h[startIndex] = sum(toXor) % q
            ch += 1

    h.reverse()
    hamming_codes.append(h)

hamming_codes = np.array(hamming_codes)
f_name = 'hamming_codes_q' + str(q) + '_n' + str(hamm_n) + '_k' + str(hamm_k) + '.txt'
np.savetxt(f_name, hamming_codes, fmt='%d')
