import {expect,test} from '@playwright/test';

test('walkthrough: screening, rejection, live risk, escalation and policy',async({page,request},testInfo)=>{
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  await request.patch('/api/policies/default/threshold',{data:{threshold:.55,onboarding_threshold:.28689560294151306}});
  const reset=await request.post('/api/scenarios/sleeper_bustout/reset');
  expect(reset.ok()).toBeTruthy();
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'Every account has a story.'})).toBeVisible();
  await expect(page.getByText('Curated replay · real models')).toBeVisible();
  await page.getByRole('button',{name:/Curated replay · real models/}).click();
  await expect(page.getByText('3/3 external bundles verified')).toBeVisible();
  await expect(page.getByText('Bundle ready · service ok')).toHaveCount(3);
  await page.screenshot({path:testInfo.outputPath('model-readiness.png'),fullPage:true});
  await page.getByRole('button',{name:/Curated replay · real models/}).click();
  await expect(page.getByRole('heading',{name:'Maya Chen'})).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('desktop-initial.png'),fullPage:true});
  await page.getByRole('button',{name:'Next event',exact:true}).click();
  await expect(page.getByText('1 of 15 events',{exact:true})).toBeVisible();
  for(let index=0;index<4;index++)await page.getByRole('button',{name:'Next event',exact:true}).click();
  await expect(page.getByRole('button',{name:/Avery Martin Screening/})).toBeVisible();
  await page.getByRole('button',{name:'Next event',exact:true}).click();
  await expect(page.getByRole('button',{name:/Avery Martin Monitoring/})).toBeVisible();
  await page.getByRole('button',{name:'Play simulation'}).click();
  await expect(page.getByRole('button',{name:/Jordan Lee Rejected at onboarding/})).toBeVisible({timeout:20000});
  await expect(page.getByRole('heading',{name:'Review required',exact:true})).toBeVisible({timeout:25000});
  await page.getByRole('button',{name:'Pause simulation'}).click();
  await expect(page.getByText('Final cash-out attempt',{exact:true})).toHaveCount(0);
  await page.getByLabel('Analyst notes').fill('Rapid repeated transfers after quiet activity. Escalate for beneficiary verification before the cash-out.');
  await page.screenshot({path:testInfo.outputPath('desktop-alert.png'),fullPage:true});
  await page.getByRole('button',{name:'Review & Escalate'}).click();
  await expect(page.getByText('Case escalated',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:/Risk policy/}).click();
  await page.getByRole('button',{name:'Growth · 85'}).click();
  await page.getByRole('button',{name:'Apply policy'}).click();
  await expect(page.getByRole('button',{name:/Risk policy 85/})).toBeVisible();
  await page.setViewportSize({width:1366,height:768});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await page.screenshot({path:testInfo.outputPath('desktop-1366.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await page.screenshot({path:testInfo.outputPath('mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'Reset simulation'}).click();
  await expect.poll(async()=>{
    const response=await request.get('/api/state');
    const state=await response.json();
    return state.scenario.cursor;
  }).toBe(0);
  expect(errors).toEqual([]);
  await request.patch('/api/policies/default/threshold',{data:{threshold:.55,onboarding_threshold:.28689560294151306}});
});

test('connection failure has an explicit state',async({page})=>{
  await page.route('**/api/state',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'API temporarily unavailable'})}));
  await page.goto('/');
  await expect(page.getByText('API temporarily unavailable',{exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'Reconnect',exact:true})).toBeVisible();
});
