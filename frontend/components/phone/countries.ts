const sc = require('states-cities-db');

const COUNTRIES = sc.getCountries();

const getCountryTelCode = (country: any) =>
  country && COUNTRIES.find(({ iso }: any) => iso === country).prefix;

export { COUNTRIES, getCountryTelCode };
