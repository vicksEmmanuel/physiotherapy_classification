import accounting from 'accounting';
import { useMemo } from 'react';
import Currencies from '../public/data/currencies.json';

export const useNumberFormatter = (currency?: string) => {
  return useMemo(() => new NumberFormatter(currency || 'USD'), [currency]);
};

export class NumberFormatter {
  constructor(private currency: string) {}

  // eslint-disable-next-line   class-methods-use-this, consistent-return
  formatNumber(
    value: number | undefined,
    size: 'integer' | 'floating' = 'floating',
    options = {
      decimals: 2,
    }
  ): string {
    if (!value) {
      value = 0;
    }

    // eslint-disable-next-line   default-case
    switch (size) {
      case 'integer':
        return accounting.formatNumber(value, 0, ',');
      case 'floating':
        return accounting.formatNumber(value, options.decimals, ',', '.');
    }
  }

  formatCurrency(value: number | string, decimals = 2) {
    if (typeof value === 'string') {
      value = parseFloat(value);
    }

    const currency = (Currencies as any)[this.currency];
    const switchOrder = currency?.switch_order;
    const symbol = currency?.symbol || this.currency;
    const number = this.formatNumber(value, 'floating', {
      decimals,
    });
    return switchOrder ? `${number} ${symbol}` : `${symbol} ${number}`;
  }
}
