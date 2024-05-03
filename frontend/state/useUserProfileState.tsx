import { Smartsheet, User, UserFragment } from '@/hooks/api/generated';
import create from 'zustand';
import { combine } from 'zustand/middleware';
import { persist } from '.';

export interface UserState {
  user: UserFragment | undefined;
  minSideBar?: boolean;
}

const defaultState: UserState = {
  user: undefined,
  minSideBar: false,
};

export const useUserProfileState = create(
  persist(
    {
      key: 'auth-client-2-user-profile-xxx-test-new-minos',
    },
    combine(defaultState, (set) => ({
      setUser: (user?: User) => set((state) => ({ ...state, user })),
      setCurrentSmartSheet: (current_smart_sheet: Smartsheet) =>
        set((state) => ({ ...state, current_smart_sheet })),
      setMinSideBar: (minSideBar: boolean) =>
        set((state) => ({ ...state, minSideBar })),
      reset: () => set(() => defaultState),
    }))
  )
);
