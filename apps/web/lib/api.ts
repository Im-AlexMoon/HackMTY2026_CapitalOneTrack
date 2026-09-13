import type {State} from './types';
export async function api(path:string,method='GET',body?:unknown):Promise<State>{
  const response=await fetch('/api'+path,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),cache:'no-store'});
  if(!response.ok){const text=await response.json().catch(()=>({detail:'Request failed'}));throw new Error(typeof text.detail==='string'?text.detail:JSON.stringify(text.detail));}
  return response.json();
}
