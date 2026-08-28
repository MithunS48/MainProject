import { Link } from "react-router-dom";
import { Fish, Mail, ShieldCheck } from "lucide-react";

export default function PublicFooter() {
  return (
    <footer className="relative border-t border-slate-200 bg-white pt-16 pb-8">
      <div className="max-w-7xl mx-auto px-5 sm:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10">
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-ocean-400 to-seaweed-500 flex items-center justify-center">
                <Fish className="w-5 h-5 text-white" />
              </div>
              <span className="font-display font-bold text-lg text-slate-900">AquaScan</span>
            </div>
            <p className="text-slate-500 text-sm max-w-sm leading-relaxed">
              AI-assisted fish disease detection for farmers and aquaculture
              professionals — powered by MobileNetV2 + ConvNeXt feature
              fusion and a Polynomial SVM classifier, trained to 98.29% test
              accuracy.
            </p>
            <div className="flex items-center gap-2 mt-4 text-xs text-slate-400">
              <ShieldCheck className="w-4 h-4" />
              AI-assisted tool — not a substitute for professional veterinary advice.
            </div>
          </div>

          <div>
            <h4 className="text-slate-900 font-semibold text-sm mb-4">Platform</h4>
            <ul className="space-y-2.5 text-sm text-slate-500">
              <li><a href="#how-it-works" className="hover:text-slate-900 transition-colors">How It Works</a></li>
              <li><a href="#technology" className="hover:text-slate-900 transition-colors">AI Technology</a></li>
              <li><a href="#diseases" className="hover:text-slate-900 transition-colors">Supported Diseases</a></li>
              <li><a href="#why-us" className="hover:text-slate-900 transition-colors">Model Information</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-slate-900 font-semibold text-sm mb-4">Account</h4>
            <ul className="space-y-2.5 text-sm text-slate-500">
              <li><Link to="/login" className="hover:text-slate-900 transition-colors">Log In</Link></li>
              <li><Link to="/register" className="hover:text-slate-900 transition-colors">Create Account</Link></li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-6 border-t border-slate-200 flex flex-col sm:flex-row justify-between items-center gap-4 text-xs text-slate-400">
          <span>&copy; {new Date().getFullYear()} AquaScan. AI-Based Fish Disease Detection and Classification System.</span>
          <div className="flex items-center gap-4">
            <Mail className="w-4 h-4" />
          </div>
        </div>
      </div>
    </footer>
  );
}
