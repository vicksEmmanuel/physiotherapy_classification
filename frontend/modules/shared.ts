export function isBrowser() {
  return typeof window !== 'undefined';
}

export function subStr(str: string, end: number) {
  return str?.substring(0, end);
}

export const UrlRegex =
  /(?:^|[^@\.\w-])([a-z0-9]+:\/\/)?(\w(?!ailto:)\w+:\w+@)?([\w.-]+\.[a-z]{2,4})(:[0-9]+)?(\/.*)?(?=$|[^@\.\w-])/im;
