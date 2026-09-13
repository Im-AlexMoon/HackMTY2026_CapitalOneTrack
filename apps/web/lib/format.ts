import type {Account,Prediction} from './types';
export const score=(v:number|null|undefined)=>typeof v==='number'&&Number.isFinite(v)?String(Math.round(v*100)):'—';
export const availableScore=(p:Prediction|null|undefined)=>p&&['ok','demo'].includes(p.status)?p.risk_score:null;
export const riskTone=(value:number|null,threshold:number)=>value===null?'muted':value>=threshold?'danger':value>=threshold-.10?'warning':'neutral';
export const clock=(value:string)=>Number.isNaN(Date.parse(value))?'—':new Date(value).toISOString().slice(11,19);
export const filterAccounts=(accounts:Account[],query:string,filter:string)=>accounts.filter(a=>(a.alias+' '+a.id).toLowerCase().includes(query.trim().toLowerCase())&&(filter==='all'||(filter==='attention'?['review_required','escalated'].includes(a.status):a.status===filter)));
export const statusLabel:Record<string,string>={screening:'Screening',monitoring:'Monitoring',rejected:'Rejected at onboarding',review_required:'Review required',escalated:'Escalated'};
export const money=(n:number,currency='USD')=>new Intl.NumberFormat('en-US',{style:'currency',currency,maximumFractionDigits:0}).format(n);
