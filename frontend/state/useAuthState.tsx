import create from 'zustand';
import { combine } from 'zustand/middleware';
import { persist } from '.';

const defaultState = {
  token: undefined as string | undefined,
  selectedYear: new Date().getFullYear(),
  link: '/',
};

export const useAuthState = create(
  persist(
    {
      key: 'auth',
    },
    combine(defaultState, (set) => ({
      setLink: (link: string) =>
        set((state) => ({
            ...state,
            link,
          })
      ),
      setToken: (token: string) =>
        set((state) => ({
          ...state,
          token,
        })),
      setSelectedYear: (selectedYear: number) =>
        set((state) => ({
          ...state,
          selectedYear,
        })),
      setUrl: (pharmDetailID: string) =>
        set((state) => ({
          ...state,
          pharmDetailID,
        })),
      reset: () => set(() => defaultState),
    }))
  )
);
