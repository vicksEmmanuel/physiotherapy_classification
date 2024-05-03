import HomeComponent from '@/components/HomeComponent';
import SmartSheetLayout from '@/components/Layout';
import Login from '@/components/Login';
import { User } from '@/hooks/api/generated';
import { useGraphql } from '@/hooks/useGraphql';
import { useUserProfileState } from '@/state/useUserProfileState';
import { isUndefined } from 'lodash';
import type {
  GetServerSidePropsContext,
  InferGetServerSidePropsType,
} from 'next';
import { getServerSession, } from 'next-auth/next';
import { getProviders, signOut, useSession } from 'next-auth/react';
import { authOptions } from './api/auth/[...nextauth]';

export default function Home({
  providers,
  user,
}: InferGetServerSidePropsType<typeof getServerSideProps>) {
  const {data} = useSession();
  return (
    <div className='bg-white h-full w-screen '>
      {!data?(
        <div className="w-full items-center justify-center h-[100%] flex flex-col bg-[#f5f8fa]">
          <Login providers={providers} />
        </div>
      ):(
        <SmartSheetLayout>
          <div className="w-full items-center justify-center h-[100%] flex flex-col bg-[#f5f8fa]">
              <HomeComponent />
          </div>
        </SmartSheetLayout>
      )}
    </div>
  );
}

export async function getServerSideProps(context: GetServerSidePropsContext) {
  const session = await getServerSession(context.req, context.res, authOptions);
  const graphql = useGraphql();

  const providers = await getProviders();

  const {setUser, user : userProfile} = useUserProfileState.getState()


  try {
      if (session) {

      if (!isUndefined(userProfile)) {
          return  {
            props: { providers: providers ?? [], user: userProfile},
          };
      }


        const url = context.resolvedUrl;

        if (['client'].includes(url)) {
          return { redirect: { destination: context.resolvedUrl } };
        }

        if (isUndefined(session?.user?.email)) {
          return {
            redirect: {
              destination: `/api/auth/signout?callbackUrl=${encodeURIComponent('/?signout=true')}`,
              permanent: false,
            },
          };
        }


        const firstName = session?.user?.name?.split(/\s/)?.[0];
        const lastName = session?.user?.name?.split(/\s/)?.[1] ?? '';

        try {
          //Update User:
          const user = await graphql.checkIfUserExist({
            email: session?.user?.email,
            isActive: true,
          });

          const userExit = user?.user?.find((i) => i.email === session?.user?.email);

          if (isUndefined(userExit)) {
            const userRegent = await graphql.InsertOrUpdateUsers({
              users: {
                email: session?.user?.email,
                isActive: true,
                profileImage: session?.user?.image,
                password: '',
                lastName,
                firstName,
                provider: 'physio'
              },
            });

            setUser({
              ...userRegent.insert_user?.returning?.[0] as User
            })
            
          } else {

            const userRegent = await graphql.InsertOrUpdateUsers({
              users: {
                email: userExit?.email,
                firstName: userExit?.firstName,
                lastName: userExit?.lastName,
                isActive: userExit?.isActive,
                password: userExit?.password,
                provider: userExit?.provider,
                profileImage: session?.user?.image,
                id: userExit?.id,
              },
            });

            setUser({
              ...userRegent.insert_user?.returning?.[0] as User
            })
          }
        }catch(e) {
            const userRegent = await graphql.InsertOrUpdateUsers({
              users: {
                email: session?.user?.email,
                isActive: true,
                profileImage: session?.user?.image,
                password: '',
                provider: 'physio',
                lastName,
                firstName,
              },
            });

            setUser({
              ...userRegent.insert_user?.returning?.[0] as User
            })
        }



      return { redirect: { destination: '/home' } };
      }
  }catch(e) {
    console.log("Error", e)
    signOut()
  }

  return {
    props: { providers: providers ?? [], user: null },
  };
}