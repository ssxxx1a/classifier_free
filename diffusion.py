import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def compare_diff_betw_a_b(a,b,decimal):
    import numpy as np
    return np.testing.assert_almost_equal(to_numpy(a),to_numpy(b), decimal=decimal)
class GaussianDiffusion(nn.Module):
    def __init__(self, dtype, model, classifier,cemblayer,betas, w, v, device):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.classifier=classifier
        self.cemblayer=cemblayer
        self.model.dtype = self.dtype
        self.betas = torch.tensor(betas,dtype=self.dtype).to(device)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.device = device
        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas)
        
        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim = 0)
        self.alphas_bar = torch.exp(self.log_alphas_bar)
        # self.alphas_bar = torch.cumprod(self.alphas, dim = 0)
        
        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1],[1,0],'constant', 0)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas)
        # self.sqrt_alphas = torch.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar)
        
        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0 = self.betas * torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.mu_coef_xt = torch.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.vars = torch.cat((self.tilde_betas[1:2],self.betas[1:]), 0)
        self.coef1 = torch.exp(-self.log_sqrt_alphas)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)
    @staticmethod
    def _extract(coef, t, x_shape):
        """
        input:
        coef : an array
        t : timestep
        x_shape : the shape of tensor x that has K dims(the value of first dim is batch size)
        output:
        a tensor of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]

        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        chosen = coef[t]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0, t):
        """
        calculate the parameters of q(x_t|x_0)
        """
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var
    
    def q_sample(self, x_0, t):
        """
        sample from q(x_t|x_0)
        """
        eps = torch.randn_like(x_0, requires_grad=False)
       
        return self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps, eps
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        calculate the parameters of q(x_{t-1}|x_t,x_0)
        """
        posterior_mean = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        neo_posterior_var = torch.exp(log_posterior_var)
        
        return posterior_mean, posterior_var_max, neo_posterior_var
    def p_mean_variance(self, x_t, t, **model_kwargs):
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
       
        model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"
        p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
        p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(self, x_t, t, eps):
        return self._extract(coef = self.sqrt_recip_alphas_bar, t = t, x_shape = x_t.shape) \
            * x_t - self._extract(coef = self.sqrt_one_minus_alphas_bar, t = t, x_shape = x_t.shape) * eps

    def _predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        return self._extract(coef = self.coef1, t = t, x_shape = x_t.shape) * x_t - \
            self._extract(coef = self.coef2, t = t, x_shape = x_t.shape) * eps
    def trainloss(self, x_0, **model_kwargs):
        """
        calculate the loss of denoising diffusion probabilistic model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t = torch.randint(self.T, size = (x_0.shape[0],), device=self.device)
        x_t, eps = self.q_sample(x_0, t)
        pred_eps = self.model(x_t, t, **model_kwargs)
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        return loss
    def sample(self, shape, **model_kwargs):
        """
        sample images from p_{theta}
        """
        print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T
        for _ in tqdm(range(self.T),dynamic_ncols=True):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        x_t = torch.clamp(x_t, -1, 1)
        print('ending sampling process...')
        return x_t
    
    def p_sample(self, x_t, t,return_all=True, **model_kwargs):
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t , t, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        if return_all:
            noise = torch.randn_like(x_t)
            noise[t <= 0] = 0 
            return mean + torch.sqrt(var) * noise
        else:
            return mean,var
    
    
    ######## ######## ######## ########增加classifier的处理方法 ######## ######## ######## ######## ########
    def calc_diff(self,x_t,t,use_classifier=True,use_sofrmax=True):
        #assert x_t.size(0)==1
        conditions=torch.arange(0,10,1).to('cuda')
        if use_classifier:
            logits=self.classifier(x_t,t)
            if use_sofrmax:
                #scores = F.log_softmax(logits, dim=-1)
                scores = F.softmax(logits, dim=-1)[0]
            else:
                scores=logits
        else:
            scores=torch.ones_like(conditions)*0.1
        
        pred_eps=scores[0]*self.model(x_t,t,self.cemblayer(conditions[0].repeat(x_t.size(0))))
        for i,label in enumerate(conditions[1:]):
            c=self.cemblayer(label.repeat(x_t.size(0)))
            pred_eps+=scores[i]*self.model(x_t,t,c)
        cemb=torch.zeros(size=(x_t.size(0),10), device = self.device)
        pred_eps_unc= self.model(x_t, t,cemb)
        return pred_eps,pred_eps_unc
    def p_mean_variance_for_compare(self, x_t, t,compare=False,**model_kwargs):
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        if not compare:
            cemb_shape = model_kwargs['cemb'].shape
            pred_eps_cond = self.model(x_t, t, **model_kwargs)
        
            model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
            pred_eps_uncond = self.model(x_t, t, **model_kwargs)
            pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
            
            assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
            assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
            assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"
            p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
            p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
       # else:
            
        return p_mean, p_var
    
    def p_sample_for_compare(self,x_t, t,return_all=True,**model_kwargs):
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance_for_compare(x_t , t)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        
        new_mean=self.condition_mean(mean,var,x_t,t,sum_type='prob',classifier_scale=1.0,y=None)
        if return_all:
            noise = torch.randn_like(x_t)
            noise[t <= 0] = 0 
            return new_mean + torch.sqrt(var) * noise
        else:
            return mean ,var
    def cond_fn(self,classifier,x, t, classifier_scale=1.0,y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
    def condition_mean(self, p_mean,p_var, x, t,sum_type='prob',classifier_scale=1.0,y=None):
        #classifer 训练时没有使用_scale_timesteps，所以直接self._scale_timesteps(t)->t
        #[probability,mean]
        if sum_type=='prob':
            y_list=torch.arange(0,10,1)
            gradient = self.cond_fn(self.classifier,x, t, classifier_scale=classifier_scale,y=y_list[0])
            for y_i in y_list[1:]:
                gradient += self.cond_fn(self.classifier,x, t, classifier_scale=classifier_scale,y=y_i)
        new_mean = (
           p_mean.float() + p_var * gradient.float()
        )
        return new_mean
    # def compare_cond_uncond_diff(self,shape,compare_t,**model_kwargs):#主函数 sample
    #     print('Start generating...')
    #     if model_kwargs == None:
    #         model_kwargs = {}
    #     x_t = torch.randn(shape, device = self.device)
    #     tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T
    #     for _ in tqdm(range(self.T),dynamic_ncols=True):
    #         tlist -= 1
    #         if not isinstance(compare_t,list):
    #             compare_t=[compare_t]
    #         if tlist[0] in compare_t:
    #             #计算新的mean,并对齐进行累加
    #             with torch.no_grad():
    #                 #x_t_with_cond=self.p_sample_for_compare(x_t, tlist,**model_kwargs)
    #                 noise = torch.randn_like(x_t)
    #                 noise[tlist[0] <= 0] = 0
    #                 mean_cond,var_mean=self.p_sample_for_compare(x_t, tlist,return_all=False,**model_kwargs)
    #                 mean_uc,var_uc=self.p_sample(x_t, tlist,return_all=False,**model_kwargs)
                    
    #                 x_t_with_cond=mean_cond+torch.sqrt(var_mean) * noise
    #                 x_t_no_cond=mean_uc+torch.sqrt(var_uc) * noise
    #                 try:
    #                     res=compare_diff(x_t_no_cond,x_t_with_cond,decimal=4)
    #                     print(res)
    #                 except Exception as e: 
    #                     print(e)
    #                 x_t=x_t_no_cond
    #         else:
    #             with torch.no_grad():
    #                 x_t = self.p_sample(x_t, tlist,**model_kwargs) #没有classifier的处理
           
    #     x_t = torch.clamp(x_t, -1, 1)
    #     print('ending sampling process...')
    #     return x_t
    def compare_cond_uncond_diff(self,shape,compare_t,clear_compare_results=False,use_classifier=True,**model_kwargs):#主函数 sample
        print('Start generating...')
        logger_list=[]
        if model_kwargs == None:
            model_kwargs = {}
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T
        for _ in tqdm(range(self.T),dynamic_ncols=True):
            tlist -= 1
            if not isinstance(compare_t,list):
                compare_t=[compare_t]
            if tlist[0] in compare_t:
                sum_eps_condition,eps_uncond=self.calc_diff(x_t,tlist,use_classifier)
                logger_list.append(abs(sum_eps_condition.cpu().numpy()-eps_uncond.cpu().numpy()).mean())
                if clear_compare_results:
                    try:
                        res=compare_diff_betw_a_b(sum_eps_condition,eps_uncond,decimal=3)
                        print('diff in time step :{} :\n  {}'.format(tlist,res))
                    except Exception as e:
                        print(e)
                
            #直接多调用一次了。。
            x_t = self.p_sample(x_t, tlist,**model_kwargs) #没有classifier的处理
           
        x_t = torch.clamp(x_t, -1, 1)
        print('ending sampling process...')
        return x_t,logger_list