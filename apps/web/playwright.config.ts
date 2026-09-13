import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'./e2e', workers:1, fullyParallel:false, timeout:90000,
  use:{baseURL:process.env.FIRSTWATCH_URL??'http://127.0.0.1:3000',viewport:{width:1440,height:1000},video:'on',screenshot:'only-on-failure',trace:'retain-on-failure'},
  reporter:[['list'],['html',{open:'never'}]],
});
