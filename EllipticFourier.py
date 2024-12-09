import numpy as np

class EllipticFourier():
    """A class to perform elliptic Fourier descriptor of a closed contour using algorithm adapted from "Frank P Kuhl, Charles R Giardina, Elliptic Fourier features of a closed contour, Computer Graphics and Image Processing, Volume 18, Issue 3, 1982, Pages 236-258."
    """
    def __init__(self):
        pass

    def forward(self, contour, N=64):
        """Calculate elliptic Fourier coefficients of a closed contour.

        Args:
            contour (array, [N, 2]): A closed contour having N points
            N (int, optional): The number of coefficients to decompose. Defaults to 64.

        Returns:
            A0, C0, coeffs: A0, C0 and coeffs. The coeffs is a [N, 4] array consisting of an, bn, cn and dn respective in each column.
        """
        # make contour closed, v0 v1 v2 ... v_M-1 v0
        contour_closed = np.concatenate([contour, contour[[0],:]], axis=0) # [M+1, 2]
        x_p = contour_closed[:,0] # [M+1]
        y_p = contour_closed[:,1] # [M+1]
        dx_p = np.diff(x_p) # [M]
        dy_p = np.diff(y_p) # [M]
        dt_p = np.sqrt(dx_p**2 + dy_p**2) # [M]
        t_p = np.concatenate([[0], np.cumsum(dt_p)], axis=0) # [M+1]
        T = t_p[-1] # total length of curve
        n = np.arange(1, N+1).reshape((-1, 1)) # [N, 1]

        # calculate A0 and C0
        A0 = 1/T*np.sum((x_p[1:]+x_p[:-1])/2*dt_p)
        C0 = 1/T*np.sum((y_p[1:]+y_p[:-1])/2*dt_p)

        # calculate coefficients
        an = T/(2*n**2*np.pi**2)*np.sum(dx_p/dt_p*(np.cos(2*n*np.pi*t_p[1:]/T)-np.cos(2*n*np.pi*t_p[:-1]/T)), axis=1, keepdims=True)
        bn = T/(2*n**2*np.pi**2)*np.sum(dx_p/dt_p*(np.sin(2*n*np.pi*t_p[1:]/T)-np.sin(2*n*np.pi*t_p[:-1]/T)), axis=1, keepdims=True)
        cn = T/(2*n**2*np.pi**2)*np.sum(dy_p/dt_p*(np.cos(2*n*np.pi*t_p[1:]/T)-np.cos(2*n*np.pi*t_p[:-1]/T)), axis=1, keepdims=True)
        dn = T/(2*n**2*np.pi**2)*np.sum(dy_p/dt_p*(np.sin(2*n*np.pi*t_p[1:]/T)-np.sin(2*n*np.pi*t_p[:-1]/T)), axis=1, keepdims=True)
        coeffs = np.concatenate([an, bn, cn, dn], axis=1)

        self.A0 = A0
        self.C0 = C0
        self.coeffs = coeffs

        return A0, C0, coeffs

    def backward(self, M=4096, modeStart=1, modeNum=None):
        """Calculated reversed elliptic Fourier from A0, C0 and coeffs.

        Args:
            M (int, optional): Number of points of reconstructed contour. Defaults to 4096.
            modeStart (int, optional): The n index of coeffs to reconstruct contour. Defaults to 1.
            modeNum (_type_, optional): The number of coeffs behind n to reconstruct contour. Defaults to None.

        Returns:
            _type_: _description_
        """
        if modeNum is None:
            modeNum = self.coeffs.shape[0]
        else:
            modeNum = min(int(modeNum), self.coeffs.shape[0])
        modeStart = min(max(modeStart, 1), self.coeffs.shape[0])

        t = np.linspace(0, 1.0, num=M, endpoint=False) # [M] M: number of points
        n = np.arange(modeStart, modeNum + 1).reshape((-1, 1)) # efficients selected to reconstruct
        A0 = self.A0
        C0 = self.C0
        an = self.coeffs[(modeStart-1):modeNum,[0]]
        bn = self.coeffs[(modeStart-1):modeNum,[1]]
        cn = self.coeffs[(modeStart-1):modeNum,[2]]
        dn = self.coeffs[(modeStart-1):modeNum,[3]]

        # reconstruct x_t and y_t
        x_t = A0 + np.sum(an*np.cos(2*n*np.pi*t)+bn*np.sin(2*n*np.pi*t), axis=0)
        y_t = C0 + np.sum(cn*np.cos(2*n*np.pi*t)+dn*np.sin(2*n*np.pi*t), axis=0)

        contour_reconstructed = np.concatenate([x_t.reshape((-1, 1)), y_t.reshape((-1, 1))], axis=1)

        return contour_reconstructed

    def normalize(self, rotation=True, scale=False, M=3600):
        """Normalize coeffs

        Args:
            rotation (bool, optional): Normalize rotation. Defaults to True.
            scale (bool, optional): Normalize scale. Defaults to False.
            M (int, optional): The spacing number when serach start point The larger the number, the more accurate of resutls. Defaults to 3600.

        Returns:
            Coeffs: The normalized coeffs is a [N, 4] array consisting of an, bn, cn and dn respective in each column.
        """
        t = np.linspace(0, 1.0, num=M, endpoint=False) # [M] M: number of points
        n = np.arange(1, self.coeffs.shape[0] + 1).reshape((-1, 1)) # [N,1] the num of efficients
        an = self.coeffs[:,[0]] # [N,1]
        bn = self.coeffs[:,[1]] # [N,1]
        cn = self.coeffs[:,[2]] # [N,1]
        dn = self.coeffs[:,[3]] # [N,1]

        # reconstruct translation invariant x_t and y_t
        x_t = np.sum(an*np.cos(2*n*np.pi*t)+bn*np.sin(2*n*np.pi*t), axis=0)
        y_t = np.sum(cn*np.cos(2*n*np.pi*t)+dn*np.sin(2*n*np.pi*t), axis=0)
        distance = np.sqrt(x_t**2+y_t**2)
        theta1 = t[np.argmax(distance)]*2*np.pi

        # get normalized coefficents
        an_ = an*np.cos(n*theta1)+bn*np.sin(n*theta1)
        bn_ = -an*np.sin(n*theta1)+bn*np.cos(n*theta1)
        cn_ = cn*np.cos(n*theta1)+dn*np.sin(n*theta1)
        dn_ = -cn*np.sin(n*theta1)+dn*np.cos(n*theta1)
        coeffs = np.concatenate([an_, bn_, cn_, dn_], axis=1)

        if rotation:
            # normalize start point
            x_0 = np.sum(an_)
            y_0 = np.sum(cn_)
            psi1 = np.arctan2(y_0, x_0)

            an__ = an_*np.cos(-psi1)-cn_*np.sin(-psi1)
            bn__ = bn_*np.cos(-psi1)-dn_*np.sin(-psi1)
            cn__ = an_*np.sin(-psi1)+cn_*np.cos(-psi1)
            dn__ = bn_*np.sin(-psi1)+dn_*np.cos(-psi1)
            coeffs = np.concatenate([an__, bn__, cn__, dn__], axis=1)

        if scale:
            s = 1/np.sqrt(coeffs[0,0]**2+coeffs[0,2]**2)
            coeffs *= s

        return coeffs