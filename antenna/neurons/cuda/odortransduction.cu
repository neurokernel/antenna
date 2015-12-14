#define g_Na 120.0
#define g_K  36.0
#define g_L  0.3
#define E_K  (-12.0)
#define E_Na 115.0
#define E_L  10.613

__global__ void hhn_model(
	int *spk,
	int num_neurons,
	%(type)s _dt,
	%(type)s* I_pre,
	%(type)s* X_1,
	%(type)s* X_2,
	%(type)s* X_3,
	%(type)s* g_V,
	%(type)s* V_prev)
{
	int cart_id = blockIdx.x * blockDim.x + threadIdx.x;

	if(cart_id >= num_neurons)
		return;
	%(type)s V = g_V[cart_id];
	%(type)s bias = 10;
	spk[cart_id] = 0;

	%(type)s a[3];

	a[0] = (10-V)/(100*(exp((10-V)/10)-1));
	X_1[cart_id] = a[0]*dt - X_1[cart_id]*(dt*(a[0] + exp(-V/80)/8) - 1);

	a[1] = (25-V)/(10*(exp((25-V)/10)-1));
	X_2[cart_id] = a[1]*dt - X_2[cart_id]*(dt*(a[1] + 4*exp(-V/18)) - 1);

	a[2] = 0.07*exp(-V/20);
	X_3[cart_id] = a[2]*dt - X_3[cart_id]*(dt*(a[2] + 1/(exp((30-V)/10)+1)) - 1);

	V = V + dt * (I_pre[cart_id]+bias - \
		(g_K * pow(X_1[cart_id], 4) * (V - E_K) + \
		g_Na * pow(X_2[cart_id], 3) * X_3[cart_id] * (V - E_Na) + \
		g_L * (V - E_L)));

	if(V_prev[cart_id] <= g_V[cart_id] && g_V[cart_id] > V)
		spk[cart_id] = 1;


	V_prev[cart_id] = g_V[cart_id];
	g_V[cart_id] = V;

	return;
}

__global__ void hhn_model_multi(
	int *spk,
	int num_neurons,
	%(type)s _dt,
	%(type)s* I_pre,
	%(type)s* X_1,
	%(type)s* X_2,
	%(type)s* X_3,
	%(type)s* g_V,
	%(type)s* V_prev)
{
	int cart_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (cart_id >= num_neurons)
		return;
	%(type)s V = g_V[cart_id];
	%(type)s bias = 10;
	spk[cart_id] = 0;

	int step = 1;
	%(type)s dt = 0.01;
	if(_dt > dt)
	    step = _dt/dt;

	%(type)s a[3];

	int spk_count = 0;

	for (int i=0; i<step; ++i) {
		V = g_V[cart_id];
		a[0] = (10-V)/(100*(exp((10-V)/10)-1));
		X_1[cart_id] = a[0]*dt - X_1[cart_id]*(dt*(a[0] + exp(-V/80)/8) - 1);

		a[1] = (25-V)/(10*(exp((25-V)/10)-1));
		X_2[cart_id] = a[1]*dt - X_2[cart_id]*(dt*(a[1] + 4*exp(-V/18)) - 1);

		a[2] = 0.07*exp(-V/20);
		X_3[cart_id] = a[2]*dt - X_3[cart_id]*(dt*(a[2] + 1/(exp((30-V)/10)+1)) - 1);

		V = V + dt * (I_pre[cart_id]+bias - \
			(g_K * pow(X_1[cart_id], 4) * (V - E_K) + \
			g_Na * pow(X_2[cart_id], 3) * X_3[cart_id] * (V - E_Na) + \
			g_L * (V - E_L)));

		if(V_prev[cart_id] <= g_V[cart_id] && g_V[cart_id] > V)
			spk[cart_id] = 1;

		V_prev[cart_id] = g_V[cart_id];
		g_V[cart_id] = V;
	}

	return;
}
